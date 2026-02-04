import os
import re
from typing import Dict, Optional

import gradio as gr
import torch
import torch.nn.functional as F
from librosa import load as libr_load
from soundfile import write as sf_write
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

from pyannote.core import Timeline, Segment



class TS_DiCoW_Pipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, diarization_pipeline, speaker_embedding_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.diarization_pipeline = diarization_pipeline
        self.speaker_embedding_model = speaker_embedding_model
        self.type = "seq2seq_whisper"


    #################################################################
    #                   DiCoW Utilities Function                    #
    #################################################################

    def get_diarization_mask(self, per_speaker_samples, audio_length):
        diarization_mask = torch.zeros(len(per_speaker_samples), audio_length)
        for i, speaker_samples in enumerate(per_speaker_samples):
            for start, end in speaker_samples:
                diarization_mask[i, round(start * 50):round(end * 50)] = 1
        return diarization_mask

    @staticmethod
    def get_stno_mask(diar_mask, s_index):
        non_target_mask = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
        non_target_mask[s_index] = False
        sil_frames = (1 - diar_mask).prod(axis=0)
        anyone_else = (1 - diar_mask[non_target_mask]).prod(axis=0)
        target_spk = diar_mask[s_index] * anyone_else
        non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)
        overlapping_speech = diar_mask[s_index] - target_spk
        stno_mask = torch.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)
        return stno_mask

    
    #################################################################
    #       Extracting clean speaker audio from mixed Audio         #
    #################################################################

    def subtract_timeline(self, timeline, other):
        """
        Return parts of `timeline` that do NOT overlap with `other`
        """
        result = Timeline()

        for seg in timeline:
            remaining = [seg]

            for oseg in other:
                new_remaining = []
                for r in remaining:
                    if not r.intersects(oseg):
                        new_remaining.append(r)
                    else:
                        # left part
                        if r.start < oseg.start:
                            new_remaining.append(
                                Segment(r.start, min(r.end, oseg.start))
                            )
                        # right part
                        if r.end > oseg.end:
                            new_remaining.append(
                                Segment(max(r.start, oseg.end), r.end)
                            )
                remaining = new_remaining

            for r in remaining:
                if r.duration > 0:
                    result.add(r)

        return result.support()


    def exclusive_speech_per_speaker(self, per_speaker_samples):
        """
        Input: list[Timeline]
        Output: list of dicts with exclusive timeline + duration
        """
        results = []

        for i, speaker_tl in enumerate(per_speaker_samples):
            # union of all other speakers
            others = Timeline()
            for j, other_tl in enumerate(per_speaker_samples):
                if i != j:
                    others = others.union(other_tl)

            exclusive = self.subtract_timeline(speaker_tl, others)

            results.append({
                "speaker_id": i,
                "exclusive_timeline": exclusive,
                "exclusive_duration": exclusive.duration()
            })

        return results
    
    def extract_audio_from_timeline(
        self,
        waveform: torch.Tensor,   # shape: (num_samples,)
        sr: int,
        timeline: Timeline
    ):
        """
        Extract and concatenate waveform segments defined by `timeline`
        """
        chunks = []

        for segment in timeline:
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)

            if end_sample > start_sample:
                chunks.append(waveform[start_sample:end_sample])

        if len(chunks) == 0:
            return torch.zeros(0)

        return torch.cat(chunks, dim=0)




    #################################################################
    #                     Whisper ASR Functions                     #
    #################################################################

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):

        mixed_audio_path = inputs["mixed_audio_path"]
        enrollment_audio_path = inputs.get("enrollment_audio_path", None)
        
        print("+--------------------------------+")
        print("|          Preprocessing         |")
        print("+--------------------------------+")

        # 1. Resample
        print("1. Resampling audio to 16kHz...")

        mixed_input_aud, sr = libr_load(mixed_audio_path, sr=16_000, mono=True)
        sf_write(mixed_audio_path, mixed_input_aud, sr, format='wav')
        print("input to diarization:", mixed_audio_path)
        print()

        generator = super().preprocess(
            mixed_audio_path, 
            chunk_length_s=chunk_length_s, 
            stride_length_s=stride_length_s
        )
        samples = next(generator)

        print("Preprocessed samples:", samples)
        print("samples['input_features'] shape:", samples['input_features'].shape)
        print()


        # 2. Diarization
        print("2. Performing diarization...")
        diarization_output = self.diarization_pipeline(mixed_audio_path)
        print("diarization output:", diarization_output)
        print()


        # 3. Per Speaker Diarisation results
        print("3. Extracting per-speaker samples...")
        per_speaker_samples = []
        for speaker in diarization_output.labels():
            per_speaker_samples.append(diarization_output.label_timeline(speaker))
        print("per_speaker_samples:", per_speaker_samples)
        print()


        # 4. Create diarization mask and stno mask
        print("4. Creating diarization masks...")
        diarization_mask = self.get_diarization_mask(per_speaker_samples, samples['input_features'].shape[-1] // 2)
        print("diarization_mask:", diarization_mask)
        print()

        
        # 5. Extract clean speaker audio from mixed
        print("5. Extracting clean exclusive speech audio...")

        waveform = torch.from_numpy(mixed_input_aud)

        exclusive_results = self.exclusive_speech_per_speaker(per_speaker_samples)

        clean_speaker_audio = {}

        for res in exclusive_results:
            spk_id = res["speaker_id"]
            exclusive_tl = res["exclusive_timeline"]

            clean_audio = self.extract_audio_from_timeline(
                waveform=waveform,
                sr=sr,
                timeline=exclusive_tl
            )

            clean_speaker_audio[spk_id] = clean_audio

            print(
                f"Speaker {spk_id}: "
                f"{clean_audio.numel() / sr:.2f}s exclusive speech"
            )

        
        # 6. Identify the target speaker in the diarisation output
        print("6. Find which speaker is the enrollment...")

        # 6.1 Get Target Speaker Embeddings
        enrollment_waveform, _ = libr_load(enrollment_audio_path, sr=16_000, mono=True)
        # # (T,) -> (1, T) -> (1, 1, T)
        enrollment_waveform = torch.from_numpy(enrollment_waveform).float().unsqueeze(0).unsqueeze(0)
        enrollment_embedding = self.speaker_embedding_model(enrollment_waveform)

        # 6.2 Compute embeddings for each speaker and get cos sim
        speaker_scores = {}
        speaker_embeddings = {}

        for spk_id, clean_audio in clean_speaker_audio.items():
            if clean_audio.numel() == 0:
                continue

            with torch.no_grad():
                clean_audio = clean_audio.float().unsqueeze(0).unsqueeze(0)
                spk_emb = self.speaker_embedding_model(clean_audio)
                speaker_embeddings[spk_id] = spk_emb

            enrollment_embedding = F.normalize(enrollment_embedding, dim=0)
            spk_emb = F.normalize(spk_emb, dim=0)
            score = F.cosine_similarity(enrollment_embedding, spk_emb).item()
            speaker_scores[spk_id] = score
            print(f"Speaker {spk_id} cosine sim with enrollment: {score:.4f}")

        # 6.3 Pick the speaker with highest similarity
        target_speaker_id = max(speaker_scores, key=speaker_scores.get)
        print(f"Selected target speaker: {target_speaker_id} (cos sim = {speaker_scores[target_speaker_id]:.4f})")
        samples["target_speaker_id"] = target_speaker_id
        #samples["speaker_embeddings"] = speaker_embeddings
        samples["speaker_scores"] = speaker_scores

        
        # 7. Create stno mask
        print("7. Creating stno masks...")
        stno_masks = []
        for i, speaker_samples in enumerate(per_speaker_samples):
            stno_mask = self.get_stno_mask(diarization_mask, i)
            stno_masks.append(stno_mask)
        print("stno_masks:", stno_masks)
        print()

        samples['stno_mask'] = torch.stack(stno_masks, axis=0).to(
            samples['input_features'].device,
            dtype=samples['input_features'].dtype
        )
        samples['input_features'] = samples['input_features'].repeat(
            len(per_speaker_samples), 
            1, 
            1
        )
        samples['attention_mask'] = torch.ones(
            samples['input_features'].shape[0], 
            samples['input_features'].shape[2],
            dtype=torch.bool, 
            device=samples['input_features'].device
        )

        if "num_frames" in samples:
            del samples["num_frames"]


        yield samples


    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):

        print("+--------------------------------+")
        print("|             Forward            |")
        print("+--------------------------------+")

        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        segment_size = model_inputs.pop("segment_size", None)
        is_last = model_inputs.pop("is_last")

        target_speaker_id = model_inputs.pop("target_speaker_id")
        speaker_scores = model_inputs.pop("speaker_scores")

        if stride is not None and segment_size is not None:
            raise ValueError("segment_size must be used only when stride is None")

        # Consume values so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # --------------------------------
        # 🔥 SELECT TARGET SPEAKER ONLY
        # --------------------------------
        inputs = inputs[target_speaker_id].unsqueeze(0)

        if attention_mask is not None:
            attention_mask = attention_mask[target_speaker_id].unsqueeze(0)

        # custom processing for Whisper timestamps and word-level timestamps
        if return_timestamps and self.type == "seq2seq_whisper":
            generate_kwargs["return_timestamps"] = return_timestamps
            if return_timestamps == "word":
                generate_kwargs["return_token_timestamps"] = True
                generate_kwargs["return_segments"] = True
            generate_kwargs["input_features"] = inputs

        tokens = self.model.generate(
            attention_mask=attention_mask,
            **generate_kwargs,
            **model_inputs,
        )
        
        out = {"tokens": tokens}
        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}



    def postprocess(
            self, model_outputs, decoder_kwargs: Optional[Dict] = None, return_timestamps=None, return_language=None
    ):
        
        print("+--------------------------------+")
        print("|          Postprocessing         |")
        print("+--------------------------------+")

        per_spk_outputs = self.tokenizer.batch_decode(
            model_outputs[0]['tokens'], decode_with_timestamps=True, skip_special_tokens=True
        )

        return {"text": None, "per_spk_outputs": per_spk_outputs}