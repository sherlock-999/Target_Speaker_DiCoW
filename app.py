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
STNO_CONFIG = PROJECT_ROOT / "DiariZen" / "stno_model" / "config.toml"
STNO_CHECKPOINT = PROJECT_ROOT / "DiariZen" / "stno_model" / "pytorch_model.bin"
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
    stno_config=STNO_CONFIG,
    stno_checkpoint=STNO_CHECKPOINT,
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


def build_offline_demo():
    demo = gr.Blocks(theme=gr.themes.Ocean())

    mf_audio = gr.Audio(sources="microphone", type="filepath", format="wav")
    mf_ref_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Reference speaker audio (optional)")

    mf_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[mf_audio, mf_ref_audio],
        outputs="text",
        title="DiCoW: Diarization-Conditioned Whisper",
        description=(
            "DiCoW (Diarization-Conditioned Whisper) enhances Whisper with diarization-aware transcription, enabling it to handle multi-speaker audio effectively. "
            "Use your microphone to transcribe audio with speaker-aware precision! "
            f"\nThis demo uses the checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}), "
            f"speaker diarization is powered by the  [{DIARIZATION_MODEL}](https://huggingface.co/{DIARIZATION_MODEL}). **Note:** CTC joint decoding is disabled."
        ),
        flagging_mode="never",
    )

    file_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources="upload", type="filepath", label="Audio file"),
            gr.Audio(sources=["microphone", "upload"], type="filepath", label="Reference speaker audio (optional)"),
        ],
        outputs="text",
        title="DiCoW: Diarization-Conditioned Whisper",
        description=(
            "DiCoW (Diarization-Conditioned Whisper) supports diarization-aware transcription for multi-speaker audio files. "
            f"Upload an audio file to experience state-of-the-art multi-speaker transcription. Demo uses the checkpoint "
            f"\nThis demo uses the checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}), "
            f"speaker diarization is powered by the  [{DIARIZATION_MODEL}](https://huggingface.co/{DIARIZATION_MODEL}). **Note:** CTC joint decoding is disabled."
        ),
        flagging_mode="never",
    )

    with demo:
        gr.TabbedInterface([file_transcribe, mf_transcribe], ["Audio file", "Microphone"])

        gr.Markdown(
            """
            ## Features

            - **Multi-Speaker ASR**: Handles multi-speaker audio using diarization-aware transcription.  
            - **Flexible Input Sources**:  
              - **Microphone**: Record and transcribe live audio.  
              - **Audio File Upload**: Upload pre-recorded audio files for transcription.  
            - **Diarization Support**: Powered by `BUT-FIT/diarizen-wavlm-large-s80-md` for accurate speaker segmentation.  
            - **Built with 🤗 Transformers**: Uses the latest Whisper checkpoints for robust transcription.  

            ## Citation
            If you use our model or code, please, cite:
            ```bibtex
            @article{POLOK2026101841,
                title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
                journal = {Computer Speech & Language},
                volume = {95},
                pages = {101841},
                year = {2026},
                issn = {0885-2308},
                doi = {https://doi.org/10.1016/j.csl.2025.101841},
                url = {https://www.sciencedirect.com/science/article/pii/S088523082500066X},
                author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
                keywords = {Diarization-conditioned Whisper, Target-speaker ASR, Speaker diarization, Long-form ASR, Whisper adaptation},
            }
            
            @INPROCEEDINGS{10887683,
              author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Černocký, Jan and Burget, Lukáš},
              booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
              title={Target Speaker ASR with Whisper}, 
              year={2025},
              volume={},
              number={},
              pages={1-5},
              keywords={Transforms;Signal processing;Transformers;Acoustics;Speech processing;target-speaker ASR;diarization conditioning;multi-speaker ASR;Whisper},
              doi={10.1109/ICASSP49660.2025.10887683}
            }
            
            ```
            ## License
            
            This project combines multiple components, each with its own license:
            
            * **DiCoW** (this repository): Licensed under the [Apache License 2.0](LICENSE).
            * **DiCoW Model Weights**: Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) – attribution required for usage.
            * **Diarizen (BUT-FIT/diarizen-wavlm-large-s80-mlc)**: Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) – free for research and non-commercial use only.

            Please ensure compliance with the respective licenses when using, modifying, or redistributing these components.
            
            ## Contributing
            We welcome contributions! If you’d like to add features or improve our pipeline, please open an issue or submit a pull request.
            """
        )
        mf_audio.start_recording(lambda: gr.Warning("Please wait for the audio to be displayed before submitting!"))

    return demo


def build_chunked_demo():
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.Markdown(
            f"""
            # DiCoW Chunked Online Target Speaker

            Upload mixed-speaker audio and an enrollment clip. Audio is split into VAD-based speech chunks; each chunk is diarized independently, the target speaker is matched via embedding similarity, and DiCoW transcribes only that speaker. Chunk-level results with confidence scores are returned.

            Unlike the streaming STNO route, this uses full offline diarization per chunk — no streaming model required.

            Checkpoint: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})  
            Diarization: [{DIARIZATION_MODEL}](https://huggingface.co/{DIARIZATION_MODEL})
            """
        )
        with gr.Row():
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                label="Mixed-speaker audio",
            )
            enrollment_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Enrollment speaker audio")
        run = gr.Button("Transcribe target speaker", variant="primary")
        transcript = gr.Textbox(label="Target-speaker transcript", lines=8)
        run.click(
            fn=transcribe_chunked_target,
            inputs=[audio, enrollment_audio],
            outputs=[transcript],
        )
        audio.start_recording(lambda: gr.Warning("Please wait for the mixed audio to be displayed before submitting!"))
    return demo


def build_online_target_demo():
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.Markdown(
            f"""
            # DiCoW Online Target Speaker

            Record or upload mixed-speaker audio and provide enrollment audio for the target speaker. This route uses streaming STNO conditioning with rolling online DiCoW decoding and returns committed target-speaker transcript chunks.

            Checkpoint: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})  
            Diarization: [{DIARIZATION_MODEL}](https://huggingface.co/{DIARIZATION_MODEL})
            """
        )
        with gr.Row():
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                label="Mixed-speaker audio",
            )
            enrollment_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Enrollment speaker audio")
        run = gr.Button("Transcribe target speaker", variant="primary")
        transcript = gr.Textbox(label="Target-speaker transcript", lines=8)
        run.click(
            fn=transcribe_online_target,
            inputs=[audio, enrollment_audio],
            outputs=[transcript],
        )
        audio.start_recording(lambda: gr.Warning("Please wait for the mixed audio to be displayed before submitting!"))
    return demo


def build_app():
    app = FastAPI()
    gr.mount_gradio_app(app, build_offline_demo().queue(max_size=5), path="/gradio-demo")
    gr.mount_gradio_app(app, build_chunked_demo().queue(max_size=2), path="/chunked-demo")
    gr.mount_gradio_app(app, build_online_target_demo().queue(max_size=2), path="/target-demo")
    return app


def build_share_demo():
    return gr.TabbedInterface(
        [build_offline_demo(), build_online_target_demo()],
        ["Offline DiCoW", "Online Target Speaker"],
    )


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
