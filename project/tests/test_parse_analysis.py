import unittest

from src.parse_brahe_analysis import parse_analysis, parse_record


class ParseAnalysisTests(unittest.TestCase):
    def test_parses_known_labels_to_snake_case_raw_keys(self):
        analysis = "\n".join(
            [
                "Summary: A short scene.",
                "Enunciation: elevated narrator",
                "Tone: ironic",
                "Genre: social novel",
                "Speech standard: formal prose",
                "Literary form: narration",
                "Active character: Alice",
                "Fuzzy time: evening",
                "Fuzzy place: drawing room",
            ]
        )

        parsed = parse_analysis(analysis)

        self.assertEqual(parsed["summary"], "A short scene.")
        self.assertEqual(parsed["enunciation_raw"], "elevated narrator")
        self.assertEqual(parsed["tone_raw"], "ironic")
        self.assertEqual(parsed["genre_raw"], "social novel")
        self.assertEqual(parsed["speech_standard_raw"], "formal prose")
        self.assertEqual(parsed["literary_form_raw"], "narration")
        self.assertEqual(parsed["active_character_raw"], "Alice")
        self.assertEqual(parsed["fuzzy_time_raw"], "evening")
        self.assertEqual(parsed["fuzzy_place_raw"], "drawing room")
        self.assertEqual(parsed["other_labels"], {})

    def test_missing_labels_are_present_as_none(self):
        parsed = parse_analysis("Genre: gothic\nTone: tense")

        self.assertIsNone(parsed["summary"])
        self.assertIsNone(parsed["enunciation_raw"])
        self.assertEqual(parsed["tone_raw"], "tense")
        self.assertEqual(parsed["genre_raw"], "gothic")
        self.assertIsNone(parsed["speech_standard_raw"])
        self.assertIsNone(parsed["literary_form_raw"])
        self.assertIsNone(parsed["active_character_raw"])
        self.assertIsNone(parsed["fuzzy_time_raw"])
        self.assertIsNone(parsed["fuzzy_place_raw"])

    def test_multiline_values_attach_to_previous_label(self):
        analysis = "Summary: First line\ncontinues here\nGenre: mystery"

        parsed = parse_analysis(analysis)

        self.assertEqual(parsed["summary"], "First line\ncontinues here")
        self.assertEqual(parsed["genre_raw"], "mystery")

    def test_unrecognized_labels_are_kept_in_other_labels(self):
        analysis = "Narrative arc: revelation\nAbsolute place: London\nGenre: detective"

        parsed = parse_analysis(analysis)

        self.assertEqual(parsed["genre_raw"], "detective")
        self.assertEqual(
            parsed["other_labels"],
            {
                "narrative_arc": "revelation",
                "absolute_place": "London",
            },
        )

    def test_parse_record_preserves_source_fields_and_adds_parsed_fields(self):
        row = {
            "instruction_id": "row-1",
            "full_text": "Excerpt text.",
            "analysis": "Enunciation: direct\nGenre: realist",
        }

        record = parse_record(row)

        self.assertEqual(record["instruction_id"], "row-1")
        self.assertEqual(record["full_text"], "Excerpt text.")
        self.assertEqual(record["analysis"], "Enunciation: direct\nGenre: realist")
        self.assertEqual(record["enunciation_raw"], "direct")
        self.assertEqual(record["genre_raw"], "realist")


if __name__ == "__main__":
    unittest.main()
