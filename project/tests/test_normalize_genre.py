import unittest

from src.normalize_genre import (
    is_clean_classification_record,
    normalize_genre,
    parse_genre_candidates,
)


class NormalizeGenreTests(unittest.TestCase):
    def test_parses_comma_and_slash_separated_candidates(self):
        self.assertEqual(
            parse_genre_candidates("Mystery/Crime Fiction, Thriller"),
            ["mystery", "crime fiction", "thriller"],
        )

    def test_maps_common_single_genres(self):
        self.assertEqual(normalize_genre("Historical fiction")["genre_norm"], "historical")
        self.assertEqual(normalize_genre("Tragedy")["genre_norm"], "drama")
        self.assertEqual(normalize_genre("Science-fiction")["genre_norm"], "speculative")
        self.assertEqual(normalize_genre("Detective fiction")["genre_norm"], "mystery_crime")

    def test_keeps_compounds_when_candidates_share_one_label(self):
        result = normalize_genre("Mystery/Crime Fiction")

        self.assertEqual(result["genre_candidates"], ["mystery", "crime fiction"])
        self.assertEqual(result["genre_norm"], "mystery_crime")
        self.assertTrue(result["genre_keep"])
        self.assertFalse(result["genre_uncertain"])

    def test_marks_conflicting_compounds_as_uncertain_drop(self):
        result = normalize_genre("Fantasy, children's literature")

        self.assertEqual(result["genre_candidates"], ["fantasy", "children s literature"])
        self.assertEqual(result["genre_norm"], "speculative")
        self.assertFalse(result["genre_keep"])
        self.assertTrue(result["genre_uncertain"])

    def test_marks_rare_or_missing_values_as_other_drop(self):
        rare = normalize_genre("Sports fiction")
        missing = normalize_genre(None)

        self.assertEqual(rare["genre_norm"], "other")
        self.assertFalse(rare["genre_keep"])
        self.assertTrue(rare["genre_uncertain"])
        self.assertEqual(missing["genre_norm"], "other")
        self.assertFalse(missing["genre_keep"])
        self.assertTrue(missing["genre_uncertain"])

    def test_clean_classification_records_only_keep_non_other_certain_rows(self):
        self.assertTrue(
            is_clean_classification_record(
                {
                    "full_text": "Text",
                    "genre_norm": "historical",
                    "genre_keep": True,
                    "genre_uncertain": False,
                }
            )
        )
        self.assertFalse(
            is_clean_classification_record(
                {"genre_norm": "other", "genre_keep": False, "genre_uncertain": True}
            )
        )
        self.assertFalse(
            is_clean_classification_record(
                {"genre_norm": "speculative", "genre_keep": False, "genre_uncertain": True}
            )
        )


if __name__ == "__main__":
    unittest.main()
