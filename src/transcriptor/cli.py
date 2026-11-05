import argparse
import sys
from pathlib import Path
from typing import Sequence

from transcriptor.errors import TranscriptorError
from transcriptor.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriptor",
        description="Transcribe archivos MP4/WAV con Lemonade Whisper y resumirlos con Gemma.",
    )
    parser.add_argument("archivo", type=Path, help="Archivo MP4 o WAV a procesar.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Carpeta de resultados (por defecto, junto al archivo de entrada).",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:13305/api/v1",
        help="URL base de la API local de Lemonade.",
    )
    parser.add_argument(
        "--model",
        default="Whisper-Large-v3-Turbo",
        help="Modelo local de transcripción.",
    )
    parser.add_argument(
        "--summary-model",
        default="gemma4-it-e4b-FLM",
        choices=("gemma4-it-e4b-FLM", "gemma4-it-12b-FLM"),
        help="Modelo Gemma local para el resumen.",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Código de idioma para Whisper (por defecto: es; usa 'auto' para detectarlo).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=300,
        help="Duración principal de cada bloque (por defecto: 300 segundos).",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=int,
        default=5,
        help="Solapamiento entre bloques para evitar cortes de contexto.",
    )
    parser.add_argument(
        "--summary-chunk-characters",
        type=int,
        default=22_000,
        help="Máximo aproximado de caracteres por resumen parcial.",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Genera TXT/SRT/JSON sin solicitar un resumen a Gemma.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    language = None if args.language.casefold() == "auto" else args.language
    try:
        run_pipeline(
            args.archivo,
            output_dir=args.output_dir,
            api_base_url=args.api_base_url,
            transcription_model=args.model,
            summary_model=args.summary_model,
            language=language,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            summary_chunk_characters=args.summary_chunk_characters,
            skip_summary=args.transcribe_only,
        )
    except TranscriptorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
