#!/usr/bin/env python3

import os
from functools import lru_cache

import torch
import torch.nn as nn

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)

from diarizen.models.module.conformer import ConformerEncoder
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model
from diarizen.models.module.wavlm_config import get_config
from diarizen.stno_utils import derive_stno_from_binary


class Model(BaseModel):
    def __init__(
        self,
        wavlm_src: str = "wavlm_base",
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        chunk_size: int = 5,
        num_channels: int = 1,
        selected_channel: int = 0,
        sample_rate: int = 16000,
        enrollment_dim: int = 256,
        conditioning: str = "film",
        enrollment_hidden: int = 256,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=2,
            max_speakers_per_frame=None,
        )
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel
        self.conditioning = conditioning

        self.wavlm_model = self.load_wavlm(wavlm_src)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)
        self.proj = nn.Linear(wavlm_feat_dim, attention_in)
        self.lnorm = nn.LayerNorm(attention_in)

        self.enroll_proj = nn.Sequential(
            nn.Linear(enrollment_dim, enrollment_hidden),
            nn.SiLU(),
            nn.Linear(enrollment_hidden, attention_in if conditioning != "film" else 2 * attention_in),
        )
        if conditioning == "concat":
            self.condition_fuse = nn.Linear(2 * attention_in, attention_in)
        else:
            self.condition_fuse = None

        self.conformer = ConformerEncoder(
            attention_in=attention_in,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            kernel_size=kernel_size,
            dropout=dropout,
            use_posi=use_posi,
            output_activate_function=output_activate_function,
        )
        self.classifier = nn.Linear(attention_in, 2)
        self.activation = nn.Identity()

    def non_wavlm_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.enroll_proj.parameters(),
            *self.conformer.parameters(),
            *self.classifier.parameters(),
            *([] if self.condition_fuse is None else list(self.condition_fuse.parameters())),
        ]

    @property
    def dimension(self) -> int:
        return 2

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        return multi_conv_num_frames(
            num_samples,
            kernel_size=[10, 3, 3, 3, 3, 2, 2],
            stride=[5, 2, 2, 2, 2, 2, 2],
            padding=[0, 0, 0, 0, 0, 0, 0],
            dilation=[1, 1, 1, 1, 1, 1, 1],
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=[10, 3, 3, 3, 3, 2, 2],
            stride=[5, 2, 2, 2, 2, 2, 2],
            dilation=[1, 1, 1, 1, 1, 1, 1],
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        return multi_conv_receptive_field_center(
            frame,
            kernel_size=[10, 3, 3, 3, 3, 2, 2],
            stride=[5, 2, 2, 2, 2, 2, 2],
            padding=[0, 0, 0, 0, 0, 0, 0],
            dilation=[1, 1, 1, 1, 1, 1, 1],
        )

    @property
    def get_rf_info(self):
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = self.receptive_field_size(num_frames=2) - receptive_field_size
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        return (
            num_frames,
            receptive_field_size / self.sample_rate,
            receptive_field_step / self.sample_rate,
        )

    def load_wavlm(self, source: str):
        if os.path.isfile(source):
            ckpt = torch.load(source, map_location="cpu")
            if "config" not in ckpt or "state_dict" not in ckpt:
                raise ValueError("Checkpoint must contain 'config' and 'state_dict'.")
            for key, value in ckpt["config"].items():
                if "prune" in key and value is not False:
                    raise ValueError(f"Pruning must be disabled. Found: {key}={value}")
            model = wavlm_model(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"], strict=False)
            return model

        config = get_config(source)
        return wavlm_model(**config)

    def wav2wavlm(self, in_wav, model):
        layer_reps, _ = model.extract_features(in_wav)
        return torch.stack(layer_reps, dim=-1)

    def condition_features(self, features: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        conditioned = self.enroll_proj(enroll)
        if self.conditioning == "film":
            gamma, beta = torch.chunk(conditioned, 2, dim=-1)
            return features * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        if self.conditioning == "add":
            return features + conditioned.unsqueeze(1)
        if self.conditioning == "concat":
            broadcast = conditioned.unsqueeze(1).expand(-1, features.shape[1], -1)
            return self.condition_fuse(torch.cat([features, broadcast], dim=-1))
        raise ValueError(f"Unsupported conditioning mode: {self.conditioning}")

    def forward(self, waveforms: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        assert waveforms.dim() == 3
        assert enroll.dim() == 2

        waveforms = waveforms[:, self.selected_channel, :]
        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)
        wavlm_feat = torch.squeeze(self.weight_sum(wavlm_feat), -1)
        outputs = self.lnorm(self.proj(wavlm_feat))
        outputs = self.condition_features(outputs, enroll)
        outputs = self.conformer(outputs)
        return self.classifier(outputs)

    def predict_stno(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        binary = (probs >= threshold).float()
        return derive_stno_from_binary(binary[..., 0], binary[..., 1])

