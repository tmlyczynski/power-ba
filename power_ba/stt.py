from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path


class SttEngineError(RuntimeError):
    pass


@dataclass
class TranscriptEvent:
    timestamp: float
    source: str
    text: str
    is_final: bool = True
    speaker: str | None = None


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


class WhisperCppSttEngine(BaseSttEngine):
    def __init__(
        self,
        model_path: str,
        binary_path: str = "whisper-cli",
        sample_rate: int = 16000,
        chunk_seconds: int = 5,
        language: str = "pl",
        threads: int = 0,
    ) -> None:
        resolved_model = model_path.strip()
        if not resolved_model:
            raise SttEngineError(
                "`whisper_cpp` backend selected but `whisper_cpp_model_path` is empty."
            )

        if not Path(resolved_model).exists():
            raise SttEngineError(
                "whisper.cpp model path does not exist. Set a valid GGUF model path in settings."
            )

        self._model_path = resolved_model
        self._binary_path = binary_path.strip() or "whisper-cli"
        self._sample_rate = sample_rate
        self._chunk_seconds = max(2, chunk_seconds)
        self._chunk_bytes = self._sample_rate * 2 * self._chunk_seconds
        self._language = language.strip() or "pl"
        self._threads = max(0, threads)

        self._source_buffers: dict[str, deque[tuple[float, bytes]]] = {}
        self._source_total_bytes: dict[str, int] = {}

    def accept_audio(self, source: str, audio_chunk: bytes, timestamp: float) -> list[TranscriptEvent]:
        queue = self._source_buffers.setdefault(source, deque())
        queue.append((timestamp, audio_chunk))
        self._source_total_bytes[source] = self._source_total_bytes.get(source, 0) + len(audio_chunk)

        events: list[TranscriptEvent] = []
        while self._source_total_bytes.get(source, 0) >= self._chunk_bytes:
            chunk_data, chunk_ts = self._pop_bytes(source, self._chunk_bytes)
            text = self._transcribe_chunk(chunk_data)
            if text:
                events.append(
                    TranscriptEvent(
                        timestamp=chunk_ts,
                        source=source,
                        text=text,
                        is_final=True,
                    )
                )
        return events

    def flush(self) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        for source in list(self._source_buffers.keys()):
            remaining = self._source_total_bytes.get(source, 0)
            if remaining <= 0:
                continue
            chunk_data, chunk_ts = self._pop_bytes(source, remaining)
            text = self._transcribe_chunk(chunk_data)
            if text:
                events.append(
                    TranscriptEvent(
                        timestamp=chunk_ts,
                        source=source,
                        text=text,
                        is_final=True,
                    )
                )
        return events

    def _pop_bytes(self, source: str, target_bytes: int) -> tuple[bytes, float]:
        queue = self._source_buffers.setdefault(source, deque())
        if target_bytes <= 0 or not queue:
            return b"", 0.0

        out = bytearray()
        chunk_ts = queue[0][0]

        while queue and len(out) < target_bytes:
            ts, piece = queue[0]
            need = target_bytes - len(out)

            if len(piece) <= need:
                out.extend(piece)
                queue.popleft()
                chunk_ts = ts
            else:
                out.extend(piece[:need])
                queue[0] = (ts, piece[need:])
                chunk_ts = ts

        consumed = len(out)
        self._source_total_bytes[source] = max(0, self._source_total_bytes.get(source, 0) - consumed)
        return bytes(out), chunk_ts

    def _transcribe_chunk(self, pcm_data: bytes) -> str:
        if not pcm_data:
            return ""

        with tempfile.TemporaryDirectory(prefix="power-ba-whisper-") as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "chunk.wav"
            out_base = temp_path / "transcript"

            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(self._sample_rate)
                handle.writeframes(pcm_data)

            command = [
                self._binary_path,
                "-m",
                self._model_path,
                "-f",
                str(wav_path),
                "-of",
                str(out_base),
                "-otxt",
                "-np",
            ]

            if self._language:
                command.extend(["-l", self._language])
            if self._threads > 0:
                command.extend(["-t", str(self._threads)])

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except FileNotFoundError as exc:
                raise SttEngineError(
                    f"whisper.cpp binary `{self._binary_path}` was not found in PATH."
                ) from exc
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                message = stderr or "whisper.cpp command failed"
                raise SttEngineError(message) from exc

            txt_path = out_base.with_suffix(".txt")
            if txt_path.exists():
                raw = txt_path.read_text(encoding="utf-8", errors="ignore")
            else:
                raw = result.stdout

        return self._normalize_transcript(raw)

    @staticmethod
    def _normalize_transcript(raw: str) -> str:
        lines: list[str] = []
        for line in raw.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue

            if cleaned.startswith("[") and "]" in cleaned:
                cleaned = cleaned.split("]", 1)[1].strip()

            if cleaned:
                lines.append(cleaned)

        return " ".join(lines).strip()


def build_stt_engine(
    backend: str,
    vosk_model_path: str,
    sample_rate: int = 16000,
    whisper_cpp_model_path: str = "",
    whisper_cpp_binary: str = "whisper-cli",
    whisper_cpp_chunk_seconds: int = 5,
    whisper_cpp_language: str = "pl",
) -> BaseSttEngine:
    if backend == "vosk":
        return VoskSttEngine(model_path=vosk_model_path, sample_rate=sample_rate)
    if backend == "whisper_cpp":
        return WhisperCppSttEngine(
            model_path=whisper_cpp_model_path,
            binary_path=whisper_cpp_binary,
            sample_rate=sample_rate,
            chunk_seconds=whisper_cpp_chunk_seconds,
            language=whisper_cpp_language,
        )
    raise SttEngineError(f"Unsupported STT backend: {backend}")
