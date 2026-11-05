import unittest

from transcriptor.pipeline import _assemble_segments


class AssembleSegmentsTests(unittest.TestCase):
    def test_overlap_segments_are_owned_by_one_chunk(self) -> None:
        chunks = {
            "0": {
                "chunk_start": 0,
                "main_start": 0,
                "main_end": 10,
                "segments": [
                    {"start": 8, "end": 10, "text": "fin del primer bloque"},
                ],
            },
            "1": {
                "chunk_start": 8,
                "main_start": 10,
                "main_end": 20,
                "segments": [
                    {"start": 0, "end": 2, "text": "duplicado del solapamiento"},
                    {"start": 2, "end": 4, "text": "inicio del segundo bloque"},
                ],
            },
        }

        assembled = _assemble_segments(chunks, chunk_count=2, duration=20)

        self.assertEqual(
            [segment["text"] for segment in assembled],
            ["fin del primer bloque", "inicio del segundo bloque"],
        )
        self.assertEqual(assembled[1]["start"], 10)

    def test_audio_timestamps_are_shifted_to_source_timeline(self) -> None:
        chunks = {
            "0": {
                "chunk_start": 4,
                "main_start": 0,
                "main_end": 10,
                "segments": [{"start": 2, "end": 3, "text": "frase"}],
            }
        }

        assembled = _assemble_segments(chunks, chunk_count=1, duration=10)

        self.assertEqual(assembled[0]["start"], 6)
        self.assertEqual(assembled[0]["end"], 7)


if __name__ == "__main__":
    unittest.main()
