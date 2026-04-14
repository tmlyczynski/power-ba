from __future__ import annotations

import queue
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class AudioChunk:
    source: str
    data: bytes
    timestamp: float


def list_pulse_sources() -> list[str]:
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    sources: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sources.append(parts[1])
    return sources


def list_monitor_sources() -> list[str]:
    return [source for source in list_pulse_sources() if _is_monitor_source(source)]


def list_mic_sources() -> list[str]:
    return [source for source in list_pulse_sources() if not _is_monitor_source(source)]


def choose_default_monitor_source(sources: list[str]) -> str | None:
    if not sources:
        return None

    default_sink = _get_default_sink_name()
    if default_sink:
        candidate = f"{default_sink}.monitor"
        if candidate in sources:
            return candidate

    if len(sources) == 1:
        return sources[0]

    return sources[0]


def choose_default_mic_source(sources: list[str]) -> str | None:
    if not sources:
        return None

    default_source = _get_default_source_name()
    if default_source and default_source in sources:
        return default_source

    if len(sources) == 1:
        return sources[0]

    return sources[0]


def _is_monitor_source(source_name: str) -> bool:
    normalized = source_name.strip().lower()
    return normalized.endswith(".monitor") or ".monitor" in normalized


def _get_default_source_name() -> str | None:
    try:
        result = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip()
    return value or None


def _get_default_sink_name() -> str | None:
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip()
    return value or None


class PulseAudioCapture:
    def __init__(
        self,
        mic_source: str,
        monitor_source: str,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
        is_source_enabled: Callable[[str], bool] | None = None,
    ) -> None:
        self.mic_source = mic_source.strip()
        self.monitor_source = monitor_source.strip()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.chunk_size_bytes = int((sample_rate * channels * 2 * chunk_ms) / 1000)
        self._is_source_enabled = is_source_enabled

        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=256)
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._wav_writers: dict[str, wave.Wave_write] = {}

    def start(self, output_dir: Path | None = None) -> None:
        self._stop_event.clear()

        if not self.monitor_source:
            raise RuntimeError("Monitor source is required (system audio source).")

        self._start_source("remote", self.monitor_source, output_dir)
        if self.mic_source:
            self._start_source("mic", self.mic_source, output_dir)

    def get_chunk(self, timeout: float = 0.25) -> AudioChunk | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()

        for proc in self._processes.values():
            if proc.poll() is None:
                proc.terminate()

        for proc in self._processes.values():
            if proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()

        for thread in self._threads:
            thread.join(timeout=1)

        for writer in self._wav_writers.values():
            writer.close()

        self._processes.clear()
        self._threads.clear()
        self._wav_writers.clear()

    def _start_source(self, name: str, source: str, output_dir: Path | None) -> None:
        command = [
            "parec",
            "-d",
            source,
            "--raw",
            "--rate",
            str(self.sample_rate),
            "--channels",
            str(self.channels),
            "--format=s16le",
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Missing `parec` command. Install pulseaudio-utils.") from exc

        if process.stdout is None:
            raise RuntimeError("Could not capture audio stream from parec.")

        self._processes[name] = process

        writer: wave.Wave_write | None = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            wav_path = output_dir / f"{name}.wav"
            writer = wave.open(str(wav_path), "wb")
            writer.setnchannels(self.channels)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            self._wav_writers[name] = writer

        thread = threading.Thread(
            target=self._reader_loop,
            args=(name, process, writer),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _reader_loop(
        self,
        name: str,
        process: subprocess.Popen[bytes],
        writer: wave.Wave_write | None,
    ) -> None:
        assert process.stdout is not None
        stream = process.stdout

        while not self._stop_event.is_set():
            data = stream.read(self.chunk_size_bytes)
            if not data:
                break

            if self._is_source_enabled is not None and not self._is_source_enabled(name):
                continue

            timestamp = time.time()
            if writer is not None:
                writer.writeframes(data)

            try:
                self._queue.put(AudioChunk(source=name, data=data, timestamp=timestamp), timeout=0.2)
            except queue.Full:
                continue
