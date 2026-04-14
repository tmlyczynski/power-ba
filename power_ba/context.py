from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time


@dataclass
class TranscriptLine:
    timestamp: float
    source: str
    text: str


class ConversationContext:
    def __init__(self, window_seconds: int = 120) -> None:
        self.window_seconds = window_seconds
        self._lines: deque[TranscriptLine] = deque()

    def add_line(self, source: str, text: str, timestamp: float | None = None) -> None:
        ts = timestamp or time()
        cleaned = text.strip()
        if not cleaned:
            return
        self._lines.append(TranscriptLine(timestamp=ts, source=source, text=cleaned))
        self._prune(now=ts)

    def recent_lines(self, now: float | None = None) -> list[TranscriptLine]:
        self._prune(now=now)
        return list(self._lines)

    def has_recent_content(self, now: float | None = None) -> bool:
        return bool(self.recent_lines(now=now))

    def render_for_prompt(self, main_prompt: str, now: float | None = None) -> str:
        lines = self.recent_lines(now=now)
        if not lines:
            return ""

        transcript_lines = []
        for line in lines:
            label = "JA" if line.source == "mic" else "MEET"
            transcript_lines.append(f"[{label}] {line.text}")

        transcript = "\n".join(transcript_lines)
        return (
            f"GLOWNY PROMPT ROLI:\n{main_prompt.strip()}\n\n"
            "KONTEKST OSTATNIEJ ROZMOWY:\n"
            f"{transcript}\n\n"
            "Na podstawie tego zaproponuj 3-5 trafnych pytan, ktore warto zadac teraz."
        )

    def _prune(self, now: float | None = None) -> None:
        current = now or time()
        cutoff = current - self.window_seconds
        while self._lines and self._lines[0].timestamp < cutoff:
            self._lines.popleft()
