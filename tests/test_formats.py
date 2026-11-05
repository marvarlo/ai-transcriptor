import unittest

from transcriptor.formats import chunk_transcript, format_timestamp


class FormatTests(unittest.TestCase):
    def test_format_timestamp_handles_hour_boundaries(self) -> None:
        self.assertEqual(format_timestamp(3661.234), "01:01:01,234")
        self.assertEqual(format_timestamp(0.0, "."), "00:00:00.000")

    def test_chunk_transcript_keeps_segments_together(self) -> None:
        segments = [
            {"start": 0, "end": 1, "text": "Primera frase."},
            {"start": 1, "end": 2, "text": "Segunda frase."},
            {"start": 2, "end": 3, "text": "Tercera frase."},
        ]

        chunks = chunk_transcript(segments, max_characters=75)

        self.assertEqual(len(chunks), 2)
        self.assertIn("Primera frase.", chunks[0])
        self.assertIn("Segunda frase.", chunks[0])
        self.assertIn("Tercera frase.", chunks[1])
