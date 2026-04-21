import unittest

from src.normalize_enunciation import is_clean_classification_record, normalize_enunciation


class NormalizeEnunciationTests(unittest.TestCase):
    def test_maps_first_person_aliases(self):
        result = normalize_enunciation("First-person narrative")

        self.assertEqual(result["enunciation_norm"], "first_person")
        self.assertFalse(result["enunciation_uncertain"])

    def test_maps_third_person_aliases(self):
        result = normalize_enunciation("Third-person omniscient narrator")

        self.assertEqual(result["enunciation_norm"], "third_person")
        self.assertFalse(result["enunciation_uncertain"])

    def test_maps_dialogue_aliases(self):
        result = normalize_enunciation("Multiple characters speaking in dialogue")

        self.assertEqual(result["enunciation_norm"], "dialogue")
        self.assertFalse(result["enunciation_uncertain"])

    def test_maps_narrative_with_dialogue_to_mixed(self):
        result = normalize_enunciation("Third-person narrative with dialogue")

        self.assertEqual(result["enunciation_norm"], "mixed")
        self.assertFalse(result["enunciation_uncertain"])

    def test_maps_letters_to_epistolary(self):
        result = normalize_enunciation("First-person letter")

        self.assertEqual(result["enunciation_norm"], "epistolary")
        self.assertFalse(result["enunciation_uncertain"])

    def test_flags_generic_speaking_values_as_uncertain_dialogue(self):
        result = normalize_enunciation("Jesus speaking to his disciples")

        self.assertEqual(result["enunciation_norm"], "dialogue")
        self.assertTrue(result["enunciation_uncertain"])

    def test_flags_missing_or_unrecognized_values(self):
        missing = normalize_enunciation(None)
        unknown = normalize_enunciation("Unclassifiable mode")

        self.assertEqual(missing["enunciation_norm"], "other")
        self.assertTrue(missing["enunciation_uncertain"])
        self.assertEqual(unknown["enunciation_norm"], "other")
        self.assertTrue(unknown["enunciation_uncertain"])

    def test_clean_classification_records_exclude_other_and_uncertain_labels(self):
        self.assertTrue(
            is_clean_classification_record(
                {"enunciation_norm": "third_person", "enunciation_uncertain": False}
            )
        )
        self.assertFalse(
            is_clean_classification_record(
                {"enunciation_norm": "other", "enunciation_uncertain": True}
            )
        )
        self.assertFalse(
            is_clean_classification_record(
                {"enunciation_norm": "dialogue", "enunciation_uncertain": True}
            )
        )


if __name__ == "__main__":
    unittest.main()
