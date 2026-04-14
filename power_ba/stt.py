from __future__ import annotations

import json
from dataclasses import dataclass


class SttEngineError(RuntimeError):
    pass


@dataclass
class TranscriptEvent:
    timestamp: float
    source: str
    text: str
    is_final: bool = True


class BaseSttEngine:
    def accept_audio(self, source: str, audio_chunk: bytes, timestamp: float) -> list[TranscriptEvent]:
        return []

    def flush(self) -> list[TranscriptEvent]:
        return []


class VoskSttEngine(BaseSttEngine):
    def __init__(self, model_path: str, sample_rate: int = 16000, emit_partials: bool = False) -> None:
        if not model_path:
            raise SttEngineError(
                "VOSK backend selected but `vosk_model_path` is empty. "
                "Set model path in settings first."
            )

        try:
            import vosk
        except ImportError as exc:
            raise SttEngineError("`vosk` package not installed.") from exc

        self._vosk = vosk
        self._model = vosk.Model(model_path)
        self._sample_rate = sample_rate
        self._emit_partials = emit_partials
        self._recognizers: dict[str, object] = {}

    def accept_audio(self, source: str, audio_chunk: bytes, timestamp: float) -> list[TranscriptEvent]:
        recognizer = self._get_recognizer(source)
        if recognizer.AcceptWaveform(audio_chunk):
            payload = json.loads(recognizer.Result())
            text = payload.get("text", "").strip()
            if text:
                return [TranscriptEvent(timestamp=timestamp, source=source, text=text, is_final=True)]
            return []

        if not self._emit_partials:
            return []

        payload = json.loads(recognizer.PartialResult())
        text = payload.get("partial", "").strip()
        if not text:
            return []
        return [TranscriptEvent(timestamp=timestamp, source=source, text=text, is_final=False)]

    def flush(self) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        for source, recognizer in self._recognizers.items():
            payload = json.loads(recognizer.FinalResult())
            text = payload.get("text", "").strip()
            if text:
                events.append(TranscriptEvent(timestamp=0.0, source=source, text=text, is_final=True))
        return events

    def _get_recognizer(self, source: str):
        recognizer = self._recognizers.get(source)
        if recognizer is None:
            recognizer = self._vosk.KaldiRecognizer(self._model, self._sample_rate)
            self._recognizers[source] = recognizer
        return recognizer


def build_stt_engine(backend: str, vosk_model_path: str, sample_rate: int = 16000) -> BaseSttEngine:
    if backend == "vosk":
        return VoskSttEngine(model_path=vosk_model_path, sample_rate=sample_rate)
    if backend == "whisper_cpp":
        raise SttEngineError("`whisper_cpp` backend is planned but not implemented yet.")
    raise SttEngineError(f"Unsupported STT backend: {backend}")
