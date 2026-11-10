"""Local speech-to-text for dictated instructions (faster-whisper, CPU int8).

The GPU is left to the Ollama LLM; a base-size Whisper model transcribes a
10 s instruction in ~1-2 s on CPU. The model downloads once from Hugging Face
on first use (~150 MB for base.en) into the standard HF cache.
"""
from __future__ import annotations

import threading

# Biases decoding toward planning vocabulary (Whisper "initial prompt").
VOCAB_PROMPT = ('Radiotherapy plan instruction. PTV, GTV, CTV, esophagus, spinal cord, heart, lungs, '
                'left lung, right lung, D95, D98, D2, Dmax, Dmean, V20, V30, V60, Gy, percent, '
                'keep PTV coverage, reduce, increase, mean dose, max dose.')

_model = None
_lock = threading.Lock()


def _get_model(cfg: dict):
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(cfg.get('model', 'base.en'), device=cfg.get('device', 'cpu'),
                                  compute_type=cfg.get('compute_type', 'int8'))
    return _model


def transcribe(audio_path: str, cfg: dict | None = None) -> str:
    cfg = cfg or {}
    model = _get_model(cfg)
    segments, _ = model.transcribe(str(audio_path), language=cfg.get('language', 'en'), beam_size=5,
                                   vad_filter=True, initial_prompt=VOCAB_PROMPT)
    return ' '.join(s.text.strip() for s in segments).strip()
