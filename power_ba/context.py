from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import localtime, strftime, time


@dataclass
class TranscriptLine:
    timestamp: float
    source: str
    speaker: str | None
    text: str


class ConversationContext:
    def __init__(self, window_seconds: int | None = 120) -> None:
        self.window_seconds = window_seconds
        self._lines: deque[TranscriptLine] = deque()
        self._all_lines: list[TranscriptLine] = []

    def set_window_seconds(self, window_seconds: int | None) -> None:
        if window_seconds is None:
            self.window_seconds = None
            return

        self.window_seconds = max(1, int(window_seconds))

    def get_window_seconds(self) -> int | None:
        return self.window_seconds

    def add_line(
        self,
        source: str,
        text: str,
        timestamp: float | None = None,
        speaker: str | None = None,
    ) -> None:
        ts = timestamp or time()
        cleaned = text.strip()
        if not cleaned:
            return
        line = TranscriptLine(timestamp=ts, source=source, speaker=speaker, text=cleaned)
        self._lines.append(line)
        self._all_lines.append(line)
        self._prune(now=ts)

    def recent_lines(self, now: float | None = None) -> list[TranscriptLine]:
        current = now or time()
        if self.window_seconds is None:
            return list(self._all_lines)

        cutoff = current - self.window_seconds
        return [line for line in self._all_lines if line.timestamp >= cutoff]

    def has_recent_content(self, now: float | None = None) -> bool:
        return bool(self.recent_lines(now=now))

    def all_lines(self) -> list[TranscriptLine]:
        return list(self._all_lines)

    def render_for_prompt(
        self,
        main_prompt: str,
        now: float | None = None,
        ai_language: str = "pl",
    ) -> str:
        lines = self.recent_lines(now=now)
        if not lines:
            return ""

        language = self._normalize_ai_language(ai_language)

        transcript_lines = []
        for line in lines:
            label = self._label_for_line(line, ai_language=language)
            transcript_lines.append(f"[{label}] {line.text}")

        transcript = "\n".join(transcript_lines)
        if language == "en":
            return (
                f"MAIN ROLE PROMPT:\n{main_prompt.strip()}\n\n"
                "RECENT CONVERSATION CONTEXT:\n"
                f"{transcript}\n\n"
                "Based on this context, propose 3-5 relevant follow-up questions to ask now."
            )

        return (
            f"GLOWNY PROMPT ROLI:\n{main_prompt.strip()}\n\n"
            "KONTEKST OSTATNIEJ ROZMOWY:\n"
            f"{transcript}\n\n"
            "Na podstawie tego zaproponuj 3-5 trafnych pytan, ktore warto zadac teraz."
        )

    def render_full_transcript(self) -> str:
        lines = self.all_lines()
        if not lines:
            return ""

        rendered: list[str] = []
        for line in lines:
            stamp = strftime("%H:%M:%S", localtime(line.timestamp))
            label = self._label_for_line(line, ai_language="pl")
            rendered.append(f"[{stamp}] [{label}] {line.text}")

        transcript = "\n".join(rendered)
        return "PELNA TRANSKRYPCJA SESJI:\n" + transcript

    def _label_for_line(self, line: TranscriptLine, ai_language: str = "pl") -> str:
        if line.source == "mic":
            return "ME" if ai_language == "en" else "JA"
        if line.speaker:
            return f"MEET:{line.speaker}"
        return "MEET"

    def _normalize_ai_language(self, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"en", "english", "angielski"}:
            return "en"
        return "pl"

    def _prune(self, now: float | None = None) -> None:
        if self.window_seconds is None:
            return

        current = now or time()
        cutoff = current - self.window_seconds
        while self._lines and self._lines[0].timestamp < cutoff:
            self._lines.popleft()
