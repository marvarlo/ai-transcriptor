import json
from pathlib import Path
from typing import Any

import httpx

from transcriptor.errors import TranscriptorError


class LemonadeClient:
    def __init__(self, base_url: str, timeout_seconds: float = 3600) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=10,
                read=timeout_seconds,
                write=300,
                pool=10,
            )
        )

    def close(self) -> None:
        self._client.close()

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: str | None,
    ) -> dict[str, Any]:
        data: dict[str, str] = {
            "model": model,
            "response_format": "verbose_json",
        }
        if language is not None:
            data["language"] = language

        try:
            with audio_path.open("rb") as audio_file:
                response = self._client.post(
                    f"{self._base_url}/audio/transcriptions",
                    data=data,
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            message = f"HTTP {exc.response.status_code}"
            if detail:
                message += f": {detail[:1000]}"
            raise TranscriptorError(
                f"Falló la transcripción del bloque {audio_path.name}: {message}"
            ) from exc
        except (OSError, httpx.HTTPError, json.JSONDecodeError) as exc:
            raise TranscriptorError(
                f"Falló la transcripción del bloque {audio_path.name}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise TranscriptorError(
                f"Lemonade devolvió una respuesta inválida para {audio_path.name}."
            )
        return payload

    def summarize(
        self,
        text: str,
        model: str,
        *,
        final: bool = False,
    ) -> str:
        if final:
            instruction = (
                "Integra los resúmenes parciales en un resumen final detallado y "
                "coherente. Conserva los temas, decisiones, datos importantes y "
                "acciones; respeta el orden temporal de entrada, elimina "
                "repeticiones y no inventes información."
            )
        else:
            instruction = (
                "Resume detalladamente esta parte de una transcripción. Identifica "
                "temas, decisiones, datos importantes y acciones. No inventes "
                "información y señala la incertidumbre cuando corresponda."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "Responde en español, con estructura clara y basándote "
                    "únicamente en el texto proporcionado."
                ),
            },
            {"role": "user", "content": f"{instruction}\n\n{text}"},
        ]
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            message = f"HTTP {exc.response.status_code}"
            if detail:
                message += f": {detail[:1000]}"
            raise TranscriptorError(
                f"Falló la generación del resumen: {message}"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise TranscriptorError(f"Falló la generación del resumen: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranscriptorError(
                "Lemonade devolvió una respuesta de resumen sin contenido válido."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise TranscriptorError(
                "Lemonade devolvió un resumen vacío; revisa el modelo y el servidor."
            )
        return content.strip()
