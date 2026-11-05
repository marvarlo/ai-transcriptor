import json
import os
import tempfile
from pathlib import Path
from typing import Any

from transcriptor.errors import TranscriptorError


def format_timestamp(seconds: float, decimal_separator: str = ",") -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02}:{minutes:02}:{whole_seconds:02}"
        f"{decimal_separator}{milliseconds:03}"
    )


def chunk_transcript(
    segments: list[dict[str, Any]],
    max_characters: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for segment in segments:
        line = f"[{format_timestamp(segment['start'], '.')}] {segment['text'].strip()}"
        if current and current_length + len(line) + 1 > max_characters:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += len(line) + 1

    if current:
        chunks.append("\n".join(current))
    return chunks


def write_transcript_files(
    output_dir: Path,
    stem: str,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    transcript_text = "\n".join(
        f"[{format_timestamp(segment['start'], '.')}] {segment['text'].strip()}"
        for segment in segments
    )
    write_text_atomic(output_dir / f"{stem}.txt", transcript_text + "\n")

    srt_blocks = [
        "\n".join(
            [
                str(index),
                (
                    f"{format_timestamp(segment['start'])} --> "
                    f"{format_timestamp(segment['end'])}"
                ),
                segment["text"].strip(),
            ]
        )
        for index, segment in enumerate(segments, start=1)
    ]
    write_text_atomic(output_dir / f"{stem}.srt", "\n\n".join(srt_blocks) + "\n")
    write_text_atomic(
        output_dir / f"{stem}.json",
        json.dumps(
            {**metadata, "segments": segments},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise TranscriptorError(f"No se pudo escribir {path}: {exc}") from exc
