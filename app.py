import argparse
import sys
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

from rolling_online_target_dicow import RollingOnlineTargetDiCoW
from online_target_dicow import OnlineTargetDiCoW
from pipeline import DiCoWPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "DiariZen"))

from diarizen.pipelines.inference import DiariZenPipeline


MODEL_NAME = "BUT-FIT/DiCoW_v3_2"
DIARIZATION_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_lower_uppercase_mapping(tokenizer):
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower_cased_token = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else "")
        else:
            lower_cased_token = token[0].lower() + token[1:]
        if lower_cased_token != token:
            lower_index = vocab.get(lower_cased_token, None)
            if lower_index is not None:
                tokenizer.upper_cased_tokens[lower_index] = index


def load_dicow_pipeline():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(MODELS_DIR),
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODELS_DIR),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODELS_DIR),
    )
    create_lower_uppercase_mapping(tokenizer)
    dicow.set_tokenizer(tokenizer)

    diar_pipeline = DiariZenPipeline.from_pretrained(
        DIARIZATION_MODEL,
        cache_dir=str(MODELS_DIR),
    ).to(device)
    diar_pipeline.embedding_batch_size = 16
    diar_pipeline.segmentation_batch_size = 16

    return DiCoWPipeline(
        dicow,
        diarization_pipeline=diar_pipeline,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )


pipeline = load_dicow_pipeline()
online_chunked_pipeline = OnlineTargetDiCoW(
    pipeline,
    min_chunk_sec=3.0,
    max_chunk_sec=8.0,
    vad_threshold=0.5,
)
online_target_pipeline = RollingOnlineTargetDiCoW(
    pipeline,
    dicow_model_name=MODEL_NAME,
    chunk_s=6.0,
    lookback_s=20.0,
    decode_window_s=30.0,
    commit_min_match_words=2,
    commit_max_current_match_start=6,
)


def transcribe(inputs, reference_audio=None):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request. "
            "Note: You might have pressed the 'Submit' button before the audio was displayed."
        )

    if reference_audio:
        text = pipeline({"audio": inputs, "reference_audio": reference_audio}, return_timestamps=True)["text"]
    else:
        text = pipeline(inputs, return_timestamps=True)["text"]
    torch.cuda.empty_cache()
    return text


def transcribe_chunked_target(audio, enrollment_audio):
    if audio is None:
        raise gr.Error("No mixed audio submitted. Please upload or record mixed-speaker audio.")
    if enrollment_audio is None:
        raise gr.Error("No enrollment audio submitted. Please upload or record reference speaker audio.")

    result = online_chunked_pipeline.transcribe(audio, enrollment_audio)
    torch.cuda.empty_cache()
    return result["text"]


def transcribe_online_target(audio, enrollment_audio):
    if audio is None:
        raise gr.Error("No mixed audio submitted. Please upload or record mixed-speaker audio.")
    if enrollment_audio is None:
        raise gr.Error("No enrollment audio submitted. Please upload or record reference speaker audio.")

    result = online_target_pipeline.transcribe(audio, enrollment_audio)
    torch.cuda.empty_cache()
    return result["text"]


def render_offline_demo():
    gr.Markdown(
        f"""
        # Offline DiCoW

        Full-utterance diarization-conditioned ASR for multi-speaker audio. Upload a file or record from the microphone. Add optional reference speaker audio when you want DiCoW to focus on one speaker.

        Checkpoint: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})  
        Diarization: [{DIARIZATION_MODEL}](https://huggingface.co/{DIARIZATION_MODEL})  
        Note: CTC joint decoding is disabled.
        """
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Audio file or microphone recording",
        )
        reference_audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Reference speaker audio (optional)",
        )
    run = gr.Button("Run Offline DiCoW ASR", variant="primary")
    transcript = gr.Textbox(label="Full transcript", lines=8)
    run.click(
        fn=transcribe,
        inputs=[audio, reference_audio],
        outputs=[transcript],
    )
    audio.start_recording(lambda: gr.Warning("Please wait for the audio to be displayed before submitting!"))
    reference_audio.start_recording(
        lambda: gr.Warning("Please wait for the reference audio to be displayed before submitting!")
    )
    with gr.Accordion("Features, citation, and license", open=False):
        gr.Markdown(
            """
            ### Features

            - **Multi-Speaker ASR**: Handles multi-speaker audio using diarization-aware transcription.
            - **Flexible Input Sources**: Record from a microphone or upload pre-recorded audio.
            - **Diarization Support**: Powered by `BUT-FIT/diarizen-wavlm-large-s80-md`.

            ### Citation
            If you use our model or code, please cite:
            ```bibtex
            @article{POLOK2026101841,
                title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
                journal = {Computer Speech & Language},
                volume = {95},
                pages = {101841},
                year = {2026},
                doi = {https://doi.org/10.1016/j.csl.2025.101841},
                author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Cernocky and Lukas Burget},
            }

            @INPROCEEDINGS{10887683,
              author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Cernocky, Jan and Burget, Lukas},
              booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
              title={Target Speaker ASR with Whisper},
              year={2025},
              pages={1-5},
              doi={10.1109/ICASSP49660.2025.10887683}
            }
            ```

            ### License
            DiCoW code is Apache 2.0, DiCoW model weights are CC BY 4.0, and Diarizen is CC BY-NC 4.0.
            """
        )


def build_offline_demo():
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        render_offline_demo()
    return demo


def render_chunked_demo():
    gr.Markdown(
        f"""
        # Chunked Online Target Speaker

        Upload mixed-speaker audio or record from the microphone, then provide an enrollment clip for the target speaker. The audio is handled as simulated streaming: it is split into VAD-based speech chunks, each chunk is diarized independently, the target speaker is matched by embedding similarity, and DiCoW transcribes that speaker chunk by chunk.

        Unlike the STNO route, this uses offline diarization inside each speech chunk and does not require the streaming STNO model.

        Checkpoint: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})  
        Diarization: [{DIARIZATION_MODEL}](https://huggingface.co/{DIARIZATION_MODEL})
        """
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Mixed-speaker audio or microphone recording",
        )
        enrollment_audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Enrollment speaker audio",
        )
    run = gr.Button("Run Chunked Online ASR", variant="primary")
    transcript = gr.Textbox(label="Chunked target-speaker transcript", lines=8)
    run.click(
        fn=transcribe_chunked_target,
        inputs=[audio, enrollment_audio],
        outputs=[transcript],
    )
    audio.start_recording(lambda: gr.Warning("Please wait for the mixed audio to be displayed before submitting!"))
    enrollment_audio.start_recording(
        lambda: gr.Warning("Please wait for the enrollment audio to be displayed before submitting!")
    )


def build_chunked_demo():
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        render_chunked_demo()
    return demo


def render_online_target_demo():
    gr.Markdown(
        f"""
        # Streaming STNO Target Speaker

        Upload mixed-speaker audio or record from the microphone, then provide enrollment audio for the target speaker. Uploaded files are processed as simulated streams: the STNO model tracks target-speaker activity over rolling context, DiCoW decodes rolling windows, and only committed target-speaker text is returned.

        Checkpoint: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})  
        STNO model: [Adnan256/streaming-target-stno-wavlm-base](https://huggingface.co/Adnan256/streaming-target-stno-wavlm-base)
        """
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Mixed-speaker audio or microphone recording",
        )
        enrollment_audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Enrollment speaker audio",
        )
    run = gr.Button("Run Streaming STNO ASR", variant="primary")
    transcript = gr.Textbox(label="Committed streaming target-speaker transcript", lines=8)
    run.click(
        fn=transcribe_online_target,
        inputs=[audio, enrollment_audio],
        outputs=[transcript],
    )
    audio.start_recording(lambda: gr.Warning("Please wait for the mixed audio to be displayed before submitting!"))
    enrollment_audio.start_recording(
        lambda: gr.Warning("Please wait for the enrollment audio to be displayed before submitting!")
    )


def build_online_target_demo():
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        render_online_target_demo()
    return demo


def build_app():
    app = FastAPI()
    gr.mount_gradio_app(app, build_offline_demo().queue(max_size=5), path="/gradio-demo")
    gr.mount_gradio_app(app, build_chunked_demo().queue(max_size=2), path="/chunked-demo")
    gr.mount_gradio_app(app, build_online_target_demo().queue(max_size=2), path="/target-demo")
    return app


def build_share_demo():
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.Markdown(
            """
            # Target Speaker DiCoW Demos

            Choose one of the three ASR modes below. Every audio input supports either upload or microphone recording.

            | Method | What it does |
            | --- | --- |
            | **Offline DiCoW** | Runs full-utterance diarization-conditioned ASR. Optional reference audio focuses the output on one speaker. |
            | **Chunked Online** | Simulates streaming by splitting audio into VAD speech chunks, diarizing each chunk, matching the target by enrollment voice, and decoding chunk by chunk. |
            | **Streaming STNO** | Simulates streaming with target-speaker activity tracking, rolling DiCoW windows, and committed transcript updates. |
            """
        )
        with gr.Tabs():
            with gr.Tab("Offline DiCoW"):
                render_offline_demo()
            with gr.Tab("Chunked Online"):
                render_chunked_demo()
            with gr.Tab("Streaming STNO"):
                render_online_target_demo()
    return demo


app = build_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DiCoW demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Launch Gradio directly with share=True.")
    args = parser.parse_args()

    if args.share:
        build_share_demo().queue(max_size=5).launch(
            server_name=args.host,
            server_port=args.port,
            share=True,
        )
    else:
        print(f"Offline DiCoW UI:              http://{args.host}:{args.port}/gradio-demo")
        print(f"Chunked online target-speaker UI: http://{args.host}:{args.port}/chunked-demo")
        print(f"Streaming STNO target-speaker UI: http://{args.host}:{args.port}/target-demo")
        uvicorn.run(app, host=args.host, port=args.port)
