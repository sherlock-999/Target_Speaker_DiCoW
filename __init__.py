from transformers import AutoConfig, AutoModelForSpeechSeq2Seq


def _try_register() -> None:
    try:
        from .dicow_config import DiCoWConfig
        from .modeling_dicow import DiCoWForConditionalGeneration
    except Exception:
        return

    try:
        AutoConfig.register("dicow", DiCoWConfig)
        AutoModelForSpeechSeq2Seq.register(DiCoWConfig, DiCoWForConditionalGeneration)
    except Exception:
        # Safe no-op when transformers registry was already populated
        # or in environments where custom classes are unavailable.
        pass


_try_register()
