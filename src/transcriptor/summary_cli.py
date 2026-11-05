import argparse
import sys
from pathlib import Path
from typing import Sequence

from transcriptor.errors import TranscriptorError
from transcriptor.summary import summarize_transcripts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resumir-transcripciones",
        description=(
            "Combina transcripciones TXT en el orden indicado y genera "
            "un resumen Markdown con Gemma local."
        ),
    )
    parser.add_argument(
        "transcripciones",
        nargs="+",
        type=Path,
        help="Archivos TXT en orden cronológico (se requieren al menos dos).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Ruta del resumen Markdown (por defecto, junto al primer TXT "
            "con el sufijo _resumen_conjunto.md)."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:13305/api/v1",
        help="URL base de la API local de Lemonade.",
    )
    parser.add_argument(
        "--model",
        default="gemma4-it-e4b-FLM",
        choices=("gemma4-it-e4b-FLM", "gemma4-it-12b-FLM"),
        help="Modelo Gemma local que genera el resumen.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=22_000,
        help="Máximo de caracteres por fragmento de transcripción.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summarize_transcripts(
            args.transcripciones,
            output_path=args.output,
            api_base_url=args.api_base_url,
            model=args.model,
            max_characters=args.max_characters,
        )
    except TranscriptorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
