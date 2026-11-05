import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcriptor.errors import TranscriptorError
from transcriptor.summary import split_text, summarize_transcripts


class SplitTextTests(unittest.TestCase):
    def test_splits_on_line_boundaries_and_limits_chunk_size(self) -> None:
        chunks = split_text("primera\nsegunda\ntercera", max_characters=15)

        self.assertEqual(chunks, ["primera\nsegunda", "tercera"])
        self.assertTrue(all(len(chunk) <= 15 for chunk in chunks))

    def test_splits_a_single_line_that_exceeds_the_limit(self) -> None:
        self.assertEqual(split_text("abcdefgh", max_characters=3), ["abc", "def", "gh"])

    def test_rejects_non_positive_chunk_size(self) -> None:
        with self.assertRaises(TranscriptorError):
            split_text("texto", max_characters=0)


class SummarizeTranscriptsTests(unittest.TestCase):
    def test_summarizes_inputs_in_order_and_writes_one_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "primero.txt"
            second = Path(temporary_directory) / "segundo.txt"
            output = Path(temporary_directory) / "resumen.md"
            first.write_text("[00:00:00.000] primera sesión\n", encoding="utf-8")
            second.write_text("[00:00:00.000] segunda sesión\n", encoding="utf-8")
            calls: list[tuple[str, bool]] = []

            class FakeClient:
                def __init__(self, _base_url: str) -> None:
                    pass

                def summarize(
                    self,
                    text: str,
                    _model: str,
                    *,
                    final: bool = False,
                ) -> str:
                    calls.append((text, final))
                    return "resumen final" if final else "resumen parcial"

                def close(self) -> None:
                    pass

            with patch("transcriptor.summary.LemonadeClient", FakeClient):
                result = summarize_transcripts(
                    [first, second],
                    output_path=output,
                )

            self.assertEqual(result, output)
            self.assertIn("primero.txt", output.read_text(encoding="utf-8"))
            self.assertIn("segundo.txt", output.read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 3)
            self.assertLess(calls[0][0].index("primero.txt"), calls[1][0].index("segunda sesión"))
            self.assertTrue(calls[-1][1])

    def test_rejects_output_that_would_overwrite_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "primero.txt"
            second = Path(temporary_directory) / "segundo.txt"
            first.write_text("primera", encoding="utf-8")
            second.write_text("segunda", encoding="utf-8")

            with self.assertRaises(TranscriptorError):
                summarize_transcripts([first, second], output_path=first)


if __name__ == "__main__":
    unittest.main()
