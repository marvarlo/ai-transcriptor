import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from transcriptor.errors import TranscriptorError
from transcriptor.formats import (
    chunk_transcript,
    format_timestamp,
    write_text_atomic,
    write_transcript_files,
)
from transcriptor.lemonade import LemonadeClient


def run_pipeline(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    api_base_url: str = "http://localhost:13305/api/v1",
    transcription_model: str = "Whisper-Large-v3-Turbo",
    summary_model: str = "gemma4-it-e4b-FLM",
    language: str | None = "es",
    chunk_seconds: int = 300,
    overlap_seconds: int = 5,
    summary_chunk_characters: int = 22_000,
    skip_summary: bool = False,
) -> Path:
    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise TranscriptorError(f"No existe el archivo de entrada: {source}")
    if source.suffix.lower() not in {".mp4", ".wav"}:
        raise TranscriptorError("El formato de entrada debe ser MP4 o WAV.")
    if chunk_seconds <= 0 or overlap_seconds < 0:
        raise TranscriptorError("La duración del bloque debe ser positiva y el "
                                "solapamiento no puede ser negativo.")
    if overlap_seconds >= chunk_seconds:
        raise TranscriptorError("El solapamiento debe ser menor que la duración "
                                "del bloque.")
    if summary_chunk_characters <= 0:
        raise TranscriptorError("El límite de caracteres del resumen debe ser "
                                "positivo.")

    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else source.parent / f"{source.stem}_transcripcion"
    )
    destination.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise TranscriptorError(
            "Se requieren ffmpeg y ffprobe en PATH para preparar el audio."
        )

    duration = _probe_duration(ffprobe, source)
    fingerprint = {
        "source": str(source),
        "size": source.stat().st_size,
        "modified_ns": source.stat().st_mtime_ns,
        "model": transcription_model,
        "language": language,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
    }
    state_path = destination / ".transcriptor-state.json"
    completed_chunks = _load_completed_chunks(state_path, fingerprint)
    chunk_count = math.ceil(duration / chunk_seconds)

    client = LemonadeClient(api_base_url)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".transcriptor-", dir=destination
        ) as temporary_dir:
            temporary_path = Path(temporary_dir)
            prepared_audio = temporary_path / "prepared.wav"
            print("Preparando audio con FFmpeg...")
            _prepare_audio(ffmpeg, source, prepared_audio)

            for index in range(chunk_count):
                key = str(index)
                if key in completed_chunks:
                    print(f"Bloque {index + 1}/{chunk_count}: recuperado del progreso.")
                    continue

                main_start = index * chunk_seconds
                main_end = min(duration, (index + 1) * chunk_seconds)
                chunk_start = max(0.0, main_start - overlap_seconds)
                chunk_end = min(duration, main_end + overlap_seconds)
                chunk_path = temporary_path / f"chunk-{index:05}.wav"
                _extract_chunk(
                    ffmpeg,
                    prepared_audio,
                    chunk_path,
                    chunk_start,
                    chunk_end - chunk_start,
                )
                print(
                    f"Transcribiendo bloque {index + 1}/{chunk_count} "
                    f"({format_timestamp(chunk_start)}–{format_timestamp(chunk_end)})."
                )
                response = client.transcribe(
                    chunk_path,
                    transcription_model,
                    language,
                )
                normalized = _normalize_transcription(response, chunk_path.name)
                completed_chunks[key] = {
                    "chunk_start": chunk_start,
                    "main_start": main_start,
                    "main_end": main_end,
                    **normalized,
                }
                write_text_atomic(
                    state_path,
                    json.dumps(
                        {"fingerprint": fingerprint, "chunks": completed_chunks},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )

        segments = _assemble_segments(completed_chunks, chunk_count, duration)
        if not segments:
            raise TranscriptorError(
                "Whisper no devolvió segmentos de voz; no se generaron resultados."
            )
        metadata = {
            "source": str(source),
            "duration_seconds": duration,
            "transcription_model": transcription_model,
            "language": language or "auto",
        }
        write_transcript_files(destination, source.stem, segments, metadata)

        if not skip_summary:
            summary_parts = chunk_transcript(segments, summary_chunk_characters)
            partial_summaries: list[str] = []
            for index, text in enumerate(summary_parts, start=1):
                print(f"Generando resumen parcial {index}/{len(summary_parts)}.")
                partial_summaries.append(
                    client.summarize(text, summary_model)
                )
            summary = (
                partial_summaries[0]
                if len(partial_summaries) == 1
                else client.summarize(
                    "\n\n".join(
                        f"Resumen parcial {index}:\n{text}"
                        for index, text in enumerate(partial_summaries, start=1)
                    ),
                    summary_model,
                    final=True,
                )
            )
            write_text_atomic(
                destination / f"{source.stem}_resumen.md",
                f"# Resumen: {source.stem}\n\n"
                f"Modelo: `{summary_model}`\n\n{summary}\n",
            )
    finally:
        client.close()

    print(f"Resultados guardados en: {destination}")
    return destination


def _probe_duration(ffprobe: str, source: Path) -> float:
    result = _run_process(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        "No se pudo leer la duración del archivo.",
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise TranscriptorError(
            f"ffprobe devolvió una duración inválida para {source}."
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise TranscriptorError(f"El archivo no contiene una duración válida: {source}")
    return duration


def _prepare_audio(ffmpeg: str, source: Path, destination: Path) -> None:
    _run_process(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        f"No se pudo extraer o convertir el audio de {source}.",
    )


def _extract_chunk(
    ffmpeg: str,
    source: Path,
    destination: Path,
    start: float,
    duration: float,
) -> None:
    _run_process(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        f"No se pudo crear el bloque de audio {destination.name}.",
    )


def _run_process(command: list[str], error_message: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise TranscriptorError(f"{error_message} {exc}") from exc
    if result.returncode != 0:
        details = result.stderr.strip()
        raise TranscriptorError(
            f"{error_message}"
            + (f"\n{details[-2000:]}" if details else "")
        )
    return result


def _load_completed_chunks(
    state_path: Path,
    fingerprint: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptorError(
            f"No se pudo leer el progreso existente {state_path}: {exc}"
        ) from exc
    if not isinstance(state, dict):
        raise TranscriptorError(f"El archivo de progreso es inválido: {state_path}")
    if state.get("fingerprint") != fingerprint:
        return {}
    chunks = state.get("chunks")
    if not isinstance(chunks, dict):
        raise TranscriptorError(f"El progreso no contiene bloques válidos: {state_path}")
    return {
        key: value
        for key, value in chunks.items()
        if isinstance(key, str)
        and isinstance(value, dict)
        and isinstance(value.get("segments"), list)
    }


def _normalize_transcription(
    response: dict[str, Any],
    chunk_name: str,
) -> dict[str, Any]:
    raw_segments = response.get("segments")
    if not isinstance(raw_segments, list):
        raise TranscriptorError(
            f"Whisper no devolvió segmentos con tiempos para {chunk_name}."
        )
    segments: list[dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise TranscriptorError(
                f"Whisper devolvió un segmento inválido para {chunk_name}."
            )
        start = item.get("start")
        end = item.get("end")
        text = item.get("text")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not isinstance(text, str)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
        ):
            raise TranscriptorError(
                f"Whisper devolvió tiempos o texto inválidos para {chunk_name}."
            )
        segments.append(
            {"start": float(start), "end": float(end), "text": text.strip()}
        )
    return {"text": response.get("text", ""), "segments": segments}


def _assemble_segments(
    chunks: dict[str, dict[str, Any]],
    chunk_count: int,
    duration: float,
) -> list[dict[str, Any]]:
    assembled: list[dict[str, Any]] = []
    for index in range(chunk_count):
        chunk = chunks.get(str(index))
        if chunk is None:
            raise TranscriptorError(
                f"Falta el resultado de transcripción del bloque {index + 1}."
            )
        offset = chunk.get("chunk_start")
        main_start = chunk.get("main_start")
        main_end = chunk.get("main_end")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (offset, main_start, main_end)
        ):
            raise TranscriptorError(
                f"El progreso guardado del bloque {index + 1} está incompleto."
            )
        segments = chunk.get("segments")
        if not isinstance(segments, list):
            raise TranscriptorError(
                f"El progreso guardado del bloque {index + 1} no contiene segmentos."
            )
        for segment in segments:
            if not isinstance(segment, dict):
                raise TranscriptorError(
                    f"El segmento guardado del bloque {index + 1} es inválido."
                )
            start = segment.get("start")
            end = segment.get("end")
            text = segment.get("text")
            if (
                isinstance(start, bool)
                or not isinstance(start, (int, float))
                or isinstance(end, bool)
                or not isinstance(end, (int, float))
                or not isinstance(text, str)
            ):
                raise TranscriptorError(
                    f"El segmento guardado del bloque {index + 1} es inválido."
                )
            if start < 0 or end < start:
                raise TranscriptorError(
                    f"El segmento guardado del bloque {index + 1} tiene tiempos "
                    "inválidos."
                )
            midpoint = (start + end) / 2
            absolute_midpoint = offset + midpoint
            if absolute_midpoint < main_start or absolute_midpoint >= main_end:
                continue
            absolute_start = max(0.0, min(duration, offset + start))
            absolute_end = max(absolute_start, min(duration, offset + end))
            if text.strip():
                assembled.append(
                    {
                        "start": absolute_start,
                        "end": absolute_end,
                        "text": text.strip(),
                    }
                )
    assembled.sort(key=lambda segment: (segment["start"], segment["end"]))
    return assembled
