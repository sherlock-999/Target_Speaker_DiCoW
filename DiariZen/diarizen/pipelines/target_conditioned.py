from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import toml
import torch
import torchaudio
from scipy.ndimage import median_filter

from huggingface_hub import snapshot_download

from diarizen.target_speaker import SpeakerEmbeddingExtractor
from diarizen.utils import instantiate
from diarizen.stno_utils import derive_stno_from_binary


class TargetConditionedSTNOPipeline:
    @staticmethod
    def _resolve_embedding_model(config: Dict, explicit_embedding_model: Optional[str]) -> str:
        if explicit_embedding_model is not None:
            return explicit_embedding_model

        for section_name in ("validate_dataset", "train_dataset"):
            section = config.get(section_name, {})
            args = section.get("args", {})
            embedding_model = args.get("embedding_model")
            if embedding_model:
                return str(embedding_model)

        return "pyannote/wespeaker-voxceleb-resnet34-LM"

    def __init__(
        self,
        diarizen_hub: Union[str, Path],
        embedding_model: Optional[str] = None,
        device: Optional[torch.device] = None,
        config_parse: Optional[Dict] = None,
    ):
        diarizen_hub = Path(diarizen_hub).expanduser().absolute()
        config = toml.load((diarizen_hub / "config.toml").as_posix())
        self._build_from_config(
            config=config,
            checkpoint_path=diarizen_hub / "pytorch_model.bin",
            embedding_model=embedding_model,
            device=device,
            config_parse=config_parse,
        )

    def _build_from_config(
        self,
        config: Dict,
        checkpoint_path: Union[str, Path],
        embedding_model: Optional[str],
        device: Optional[torch.device],
        config_parse: Optional[Dict] = None,
    ) -> None:
        if config_parse is not None:
            config["inference"]["args"] = config_parse["inference"]["args"]

        self.config = config
        self.inference_config = config.get("inference", {}).get("args", {})
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        self.model = instantiate(config["model"]["path"], args=config["model"]["args"])
        state_dict = torch.load(Path(checkpoint_path).as_posix(), map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device)

        resolved_embedding_model = self._resolve_embedding_model(config, embedding_model)
        self.embedding_extractor = SpeakerEmbeddingExtractor(
            embedding_model=resolved_embedding_model,
            device=self.device,
        )
        self.embedding_model = resolved_embedding_model
        self.frame_count, self.frame_duration, self.frame_step = self.model.get_rf_info
        self.sample_rate = self.model.sample_rate
        self.chunk_s = float(self.inference_config.get("chunk_s", config["model"]["args"]["chunk_size"]))
        self.chunk_samples = int(round(self.chunk_s * self.sample_rate))
        self.head_threshold = float(self.inference_config.get("head_threshold", 0.5))
        self.median_filter_width = int(self.inference_config.get("median_filter_width", 11))

    @classmethod
    def from_pretrained(cls, repo_id: str, cache_dir: Optional[str] = None, **kwargs):
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None,
        )
        return cls(diarizen_hub=diarizen_hub, **kwargs)

    @classmethod
    def from_experiment(
        cls,
        config_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        embedding_model: Optional[str] = None,
        device: Optional[torch.device] = None,
        config_parse: Optional[Dict] = None,
    ):
        instance = cls.__new__(cls)
        instance._build_from_config(
            config=toml.load(Path(config_path).expanduser().absolute().as_posix()),
            checkpoint_path=Path(checkpoint_path).expanduser().absolute(),
            embedding_model=embedding_model,
            device=device,
            config_parse=config_parse,
        )
        return instance

    def load_audio(self, audio: Union[str, torch.Tensor], sample_rate: Optional[int] = None) -> torch.Tensor:
        if isinstance(audio, str):
            waveform, loaded_sr = torchaudio.load(audio)
        else:
            if sample_rate is None:
                raise ValueError("`sample_rate` must be provided when passing an in-memory waveform.")
            waveform = audio
            loaded_sr = sample_rate

        waveform = waveform[:1].float()
        if loaded_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, loaded_sr, self.sample_rate)
        return waveform

    def compute_enrollment(
        self,
        reference_audio: Optional[Union[str, torch.Tensor]] = None,
        reference_sample_rate: Optional[int] = None,
        embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if embedding is not None:
            return embedding.float().to(self.device)
        if reference_audio is None:
            raise ValueError("Either `reference_audio` or `embedding` must be provided.")
        waveform = self.load_audio(reference_audio, reference_sample_rate)
        return self.embedding_extractor.embed_waveform(waveform, self.sample_rate).to(self.device)

    def infer_chunk_logits(self, waveform: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] < self.chunk_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.chunk_samples - waveform.shape[-1]))
        with torch.inference_mode():
            return self.model(waveform.unsqueeze(0).to(self.device), enrollment.unsqueeze(0)).squeeze(0).cpu()

    def finalize_logits(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = torch.sigmoid(logits)
        if self.median_filter_width > 1:
            probs = torch.from_numpy(
                median_filter(
                    probs.numpy(),
                    size=(self.median_filter_width, 1),
                    mode="reflect",
                )
            ).float()
        binary = (probs >= self.head_threshold).float()
        stno = derive_stno_from_binary(binary[..., 0], binary[..., 1]).T.contiguous()
        return {"logits": logits, "probs": probs, "binary": binary, "stno": stno}

    def infer_offline(
        self,
        mixed_audio: Union[str, torch.Tensor],
        reference_audio: Optional[Union[str, torch.Tensor]] = None,
        embedding: Optional[torch.Tensor] = None,
        mixed_sample_rate: Optional[int] = None,
        reference_sample_rate: Optional[int] = None,
        step_s: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        waveform = self.load_audio(mixed_audio, mixed_sample_rate)
        enrollment = self.compute_enrollment(reference_audio, reference_sample_rate, embedding)
        step_s = float(step_s or self.inference_config.get("segmentation_step_s", self.chunk_s))
        step_frames = max(1, int(round(step_s / self.frame_step)))

        total_frames = int(np.ceil(waveform.shape[-1] / self.sample_rate / self.frame_step))
        logits_sum = torch.zeros(total_frames, 2)
        logits_count = torch.zeros(total_frames, 1)

        for start_frame in range(0, total_frames, step_frames):
            start_sample = int(round(start_frame * self.frame_step * self.sample_rate))
            chunk = waveform[:, start_sample : start_sample + self.chunk_samples]
            chunk_logits = self.infer_chunk_logits(chunk, enrollment)
            end_frame = min(total_frames, start_frame + chunk_logits.shape[0])
            used = end_frame - start_frame
            logits_sum[start_frame:end_frame] += chunk_logits[:used]
            logits_count[start_frame:end_frame] += 1.0

        logits = logits_sum / torch.clamp(logits_count, min=1.0)
        return self.finalize_logits(logits)

    def infer_streaming(
        self,
        mixed_audio: Union[str, torch.Tensor],
        reference_audio: Optional[Union[str, torch.Tensor]] = None,
        embedding: Optional[torch.Tensor] = None,
        mixed_sample_rate: Optional[int] = None,
        reference_sample_rate: Optional[int] = None,
        chunk_s: float = 5.0,
        lookback_s: float = 20.0,
    ) -> Dict[str, torch.Tensor]:
        waveform = self.load_audio(mixed_audio, mixed_sample_rate)
        enrollment = self.compute_enrollment(reference_audio, reference_sample_rate, embedding)
        total_frames = int(np.ceil(waveform.shape[-1] / self.sample_rate / self.frame_step))
        logits = torch.zeros(total_frames, 2)

        hop_samples = int(round(chunk_s * self.sample_rate))
        lookback_samples = int(round(lookback_s * self.sample_rate))

        for global_end in range(hop_samples, waveform.shape[-1] + hop_samples, hop_samples):
            buffer_end = min(global_end, waveform.shape[-1])
            buffer_start = max(0, buffer_end - lookback_samples)
            buffer = waveform[:, buffer_start:buffer_end]
            buffer_logits = self.infer_offline(
                buffer,
                embedding=enrollment,
                mixed_sample_rate=self.sample_rate,
                step_s=chunk_s,
            )["logits"]

            global_start_frame = int(round(buffer_start / self.sample_rate / self.frame_step))
            global_end_frame = min(total_frames, global_start_frame + buffer_logits.shape[0])
            used = global_end_frame - global_start_frame
            logits[global_start_frame:global_end_frame] = buffer_logits[:used]

        return self.finalize_logits(logits)
