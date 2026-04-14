from __future__ import annotations

from dataclasses import dataclass


class LlmClient:
    def generate_questions(self, prompt_payload: str) -> str:
        raise NotImplementedError


@dataclass
class DisabledLlmClient(LlmClient):
    reason: str

    def generate_questions(self, prompt_payload: str) -> str:
        return f"[AI disabled] {self.reason}"


class OpenAiClient(LlmClient):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is empty.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("`openai` package is not installed.") from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate_questions(self, prompt_payload: str) -> str:
        try:
            response = self._client.responses.create(
                model=self._model,
                input=[
                    {
                        "role": "system",
                        "content": "Przygotuj konkretne pytania doprecyzowujace. Odpowiedz po polsku.",
                    },
                    {"role": "user", "content": prompt_payload},
                ],
            )
        except Exception as exc:  # pragma: no cover
            return f"[AI error] OpenAI request failed: {exc}"

        text = getattr(response, "output_text", "")
        if text and text.strip():
            return text.strip()

        output_items = getattr(response, "output", [])
        extracted: list[str] = []
        for item in output_items:
            for content in getattr(item, "content", []):
                maybe_text = getattr(content, "text", "")
                if maybe_text:
                    extracted.append(maybe_text)
        return "\n".join(extracted).strip() or "[AI] Brak odpowiedzi modelu."


class AnthropicClient(LlmClient):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("Anthropic API key is empty.")

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ValueError("`anthropic` package is not installed.") from exc

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def generate_questions(self, prompt_payload: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=400,
                system="Przygotuj konkretne pytania doprecyzowujace. Odpowiedz po polsku.",
                messages=[{"role": "user", "content": prompt_payload}],
            )
        except Exception as exc:  # pragma: no cover
            return f"[AI error] Anthropic request failed: {exc}"

        extracted: list[str] = []
        for item in response.content:
            text = getattr(item, "text", "")
            if text:
                extracted.append(text)

        return "\n".join(extracted).strip() or "[AI] Brak odpowiedzi modelu."


def build_llm_client(
    provider: str,
    model: str,
    openai_api_key: str,
    anthropic_api_key: str,
) -> LlmClient:
    try:
        if provider == "anthropic":
            return AnthropicClient(api_key=anthropic_api_key, model=model)
        return OpenAiClient(api_key=openai_api_key, model=model)
    except ValueError as exc:
        return DisabledLlmClient(reason=str(exc))
