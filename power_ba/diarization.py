from __future__ import annotations

import tempfile
import threading
import time
import wave
from pathlib import Path


class SpeakerDiarizationError(RuntimeError):
    pass


class BaseSpeakerDiarizer:
    status_message: str = ""

    def add_audio(self, audio_chunk: bytes, timestamp: float) -> None:
        return

    def run_if_due(self, now: float | None = None) -> None:
        return

    def label_for_timestamp(self, timestamp: float) -> str:
        return "REMOTE"

    def known_speakers(self) -> list[str]:
        return []


class NoopSpeakerDiarizer(BaseSpeakerDiarizer):
    def __init__(self, status_message: str = "Diarization disabled.") -> None:
        self.status_message = status_message


class PyannoteSpeakerDiarizer(BaseSpeakerDiarizer):
    def __init__(
        self,
        hf_token: str,
        model_name: str,
        sample_rate: int = 16000,
        run_interval_seconds: int = 15,
        max_buffer_seconds: int = 180,
        min_buffer_seconds: int = 5,
    ) -> None:
        token = hf_token.strip()
        if not token:
            raise SpeakerDiarizationError("pyannote token is empty. Set `pyannote_hf_token`.")

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise SpeakerDiarizationError(
                "`pyannote.audio` package is not installed. Install optional diarization dependencies."
            ) from exc

        self._pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        self._sample_rate = sample_rate
        self._run_interval_seconds = max(5, run_interval_seconds)
        self._max_buffer_seconds = max(30, max_buffer_seconds)
        self._min_buffer_seconds = max(2, min_buffer_seconds)

        self._buffer = bytearray()
        self._buffer_start_ts: float | None = None
        self._segments: list[tuple[float, float, str]] = []
        self._known: set[str] = set()
        self._next_run_at = 0.0
        self._lock = threading.Lock()
        self.status_message = "pyannote diarization active."

    def add_audio(self, audio_chunk: bytes, timestamp: float) -> None:
        if not audio_chunk:
            return

        duration = len(audio_chunk) / (self._sample_rate * 2)
        chunk_start = timestamp - duration

        with self._lock:
            if self._buffer_start_ts is None:
                self._buffer_start_ts = chunk_start
            self._buffer.extend(audio_chunk)
            self._trim_buffer_locked()

    def run_if_due(self, now: float | None = None) -> None:
        current = now or time.time()

        with self._lock:
            if current < self._next_run_at:
                return
            self._next_run_at = current + self._run_interval_seconds

            if self._buffer_start_ts is None:
                return

            if len(self._buffer) < int(self._sample_rate * 2 * self._min_buffer_seconds):
                return

            pcm_data = bytes(self._buffer)
            buffer_start_ts = self._buffer_start_ts

        try:
            segments = self._run_pipeline(pcm_data=pcm_data, buffer_start_ts=buffer_start_ts)
        except Exception as exc:  # pragma: no cover
            self.status_message = f"pyannote diarization error: {exc}"
            return

        with self._lock:
            self._segments = segments
            self._known = {speaker for _, _, speaker in segments}

    def label_for_timestamp(self, timestamp: float) -> str:
        with self._lock:
            for start, end, speaker in self._segments:
                if start <= timestamp <= end:
                    return speaker
        return "REMOTE"

    def known_speakers(self) -> list[str]:
        with self._lock:
            return sorted(self._known)

    def _trim_buffer_locked(self) -> None:
        max_bytes = int(self._sample_rate * 2 * self._max_buffer_seconds)
        if len(self._buffer) <= max_bytes:
            return

        to_drop = len(self._buffer) - max_bytes
        if to_drop <= 0:
            return

        del self._buffer[:to_drop]

        if self._buffer_start_ts is not None:
            self._buffer_start_ts += to_drop / (self._sample_rate * 2)

        if self._buffer_start_ts is not None:
            cutoff = self._buffer_start_ts
            self._segments = [segment for segment in self._segments if segment[1] >= cutoff]

    def _run_pipeline(self, pcm_data: bytes, buffer_start_ts: float) -> list[tuple[float, float, str]]:
        with tempfile.TemporaryDirectory(prefix="power-ba-diar-") as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "remote.wav"

            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(self._sample_rate)
                handle.writeframes(pcm_data)

            annotation = self._pipeline(str(wav_path))

        segments: list[tuple[float, float, str]] = []
        for region, _, speaker in annotation.itertracks(yield_label=True):
            abs_start = buffer_start_ts + float(region.start)
            abs_end = buffer_start_ts + float(region.end)
            segments.append((abs_start, abs_end, str(speaker)))

        segments.sort(key=lambda item: item[0])
        return segments


def build_speaker_diarizer(
    enabled: bool,
    backend: str,
    hf_token: str,
    model_name: str,
    sample_rate: int,
    interval_seconds: int,
    max_buffer_seconds: int,
) -> BaseSpeakerDiarizer:
    if not enabled:
        return NoopSpeakerDiarizer("Diarization disabled in config.")

    if backend != "pyannote":
        return NoopSpeakerDiarizer(f"Unsupported diarization backend: {backend}")

    try:
        return PyannoteSpeakerDiarizer(
            hf_token=hf_token,
            model_name=model_name,
            sample_rate=sample_rate,
            run_interval_seconds=interval_seconds,
            max_buffer_seconds=max_buffer_seconds,
        )
    except SpeakerDiarizationError as exc:
        return NoopSpeakerDiarizer(f"Diarization fallback to noop: {exc}")
