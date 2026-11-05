from pathlib import Path

from transcriptor.errors import TranscriptorError
from transcriptor.formats import write_text_atomic
from transcriptor.lemonade import LemonadeClient


def split_text(text: str, max_characters: int) -> list[str]:
    if max_characters <= 0:
        raise TranscriptorError("El límite de caracteres debe ser positivo.")

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for line in text.splitlines():
        remaining = line
        while len(remaining) > max_characters:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            chunks.append(remaining[:max_characters])
            remaining = remaining[max_characters:]

        if current and current_length + len(remaining) + 1 > max_characters:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        separator_length = 1 if current else 0
        if remaining:
            current.append(remaining)
            current_length += len(remaining) + separator_length

    if current:
        chunks.append("\n".join(current))
    return chunks


def summarize_transcripts(
    input_paths: list[Path],
    *,
    output_path: Path | None = None,
    api_base_url: str = "http://localhost:13305/api/v1",
    model: str = "gemma4-it-e4b-FLM",
    max_characters: int = 22_000,
) -> Path:
    if len(input_paths) < 2:
        raise TranscriptorError(
            "Indica al menos dos transcripciones, en el orden en que deben unirse."
        )
    if max_characters <= 0:
        raise TranscriptorError("El límite de caracteres debe ser positivo.")

    transcripts: list[tuple[Path, str]] = []
    for input_path in input_paths:
        path = input_path.expanduser().resolve()
        if not path.is_file():
            raise TranscriptorError(f"No existe la transcripción: {path}")
        if path.suffix.casefold() != ".txt":
            raise TranscriptorError(f"La transcripción debe ser un archivo TXT: {path}")
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError) as exc:
            raise TranscriptorError(
                f"No se pudo leer la transcripción {path}: {exc}"
            ) from exc
        if not text:
            raise TranscriptorError(f"La transcripción está vacía: {path}")
        transcripts.append((path, text))

    destination = (
        output_path.expanduser().resolve()
        if output_path is not None
        else transcripts[0][0].with_name(
            f"{transcripts[0][0].stem}_resumen_conjunto.md"
        )
    )
    if destination in {path for path, _ in transcripts}:
        raise TranscriptorError(
            "El archivo de salida no puede sobrescribir una de las transcripciones."
        )

    total_parts = sum(
        len(split_text(text, max_characters)) for _, text in transcripts
    )
    partial_summaries: list[str] = []
    client = LemonadeClient(api_base_url)
    try:
        part_number = 0
        for transcript_index, (path, text) in enumerate(transcripts, start=1):
            parts = split_text(text, max_characters)
            for transcript_part, part in enumerate(parts, start=1):
                part_number += 1
                print(
                    f"Resumiendo archivo {transcript_index}/{len(transcripts)}, "
                    f"parte {transcript_part}/{len(parts)} "
                    f"({part_number}/{total_parts}): {path.name}"
                )
                partial_summaries.append(
                    client.summarize(
                        f"Transcripción {transcript_index} de "
                        f"{len(transcripts)}: {path.name}\n"
                        f"Fragmento {transcript_part} de {len(parts)}.\n\n{part}",
                        model,
                    )
                )

        combined_summaries = "\n\n".join(
            f"Resumen parcial {index} (en orden de entrada):\n{summary}"
            for index, summary in enumerate(partial_summaries, start=1)
        )
        print("Integrando los resúmenes parciales.")
        final_summary = client.summarize(
            "Los siguientes resúmenes corresponden a varias transcripciones "
            "consecutivas, presentadas en orden cronológico. Integra sus temas, "
            "decisiones, datos importantes y acciones en un único resumen "
            "coherente; conserva los cambios entre sesiones cuando sean "
            "relevantes y no inventes continuidad que no aparezca en el texto.\n\n"
            f"{combined_summaries}",
            model,
            final=True,
        )
    finally:
        client.close()

    source_list = "\n".join(f"- `{path.name}`" for path, _ in transcripts)
    markdown = (
        "# Resumen integrado\n\n"
        "## Transcripciones de origen\n\n"
        f"{source_list}\n\n"
        f"Modelo: `{model}`\n\n"
        "## Resumen\n\n"
        f"{final_summary}\n"
    )
    write_text_atomic(destination, markdown)
    print(f"Resumen guardado en: {destination}")
    return destination
