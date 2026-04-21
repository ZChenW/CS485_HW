• Implemented step 1.

Parser code is in src/parse_brahe_analysis.py:46. Tests are in tests/test_parse_analysis.py:6. The parsed dataset was saved as data/
brahe_novels_parsed.jsonl:1 with 8226 records.

Example parsed fields:

{
"instruction_id": "1944_123775_1",
"summary": "The story is about an old widow and her sons, particularly one son named Vasya who breeds pigeons and sells them for extra money.",
"enunciation_raw": "Third-person narrative",
"tone_raw": "Neutral",
"genre_raw": "Realistic fiction",
"speech_standard_raw": "Conversational",
"literary_form_raw": "Description of a character and their background",
"active_character_raw": "Matvey Savitch, Marfa Semyonovna Kapluntsev, Vasya",
"fuzzy_time_raw": "Ten years ago",
"fuzzy_place_raw": "Little house, tallow and oil factory",
"other_labels": {}
}

{
"instruction_id": "24729_75592_0",
"genre_raw": "Fiction",
"fuzzy_time_raw": null,
"fuzzy_place_raw": null,
"other_labels": {
"literary_movement": "Realism"
}
}

Verification run:

python -m unittest tests.test_parse_analysis
.venv/bin/python src/parse_brahe_analysis.py --dataset-path data/brahe_novels --output data/brahe_novels_parsed.jsonl --examples 3
wc -l data/brahe_novels_parsed.jsonl

Results: 5 tests passed, conversion wrote 8226 records, output file has 8226 lines.

• Implemented step 2.

Code is in src/normalize_enunciation.py:69. Tests are in tests/test_normalize_enunciation.py:6.

Generated files:

- data/enunciation_raw_frequencies.csv:1
- data/brahe_novels_enunciation_normalized.jsonl:1
- data/enunciation_uncertain_cases.csv:1
- data/brahe_enunciation_classification.csv:1

Most common raw values:

4231 Third-person narrative
1910 First-person narrative
772 Dialog
468 Dialogue
81 Multiple characters speaking in dialogue
78 missing/blank
62 Third-person narrator
42 Characters speaking in dialogue
33 Dialogue between characters
27 Dialogue between two characters
20 Third-person narrative with dialogue
20 Conversational dialogue

First-pass normalized counts:

third_person 4329
first_person 1927
dialogue 1753
other 136
mixed 63
epistolary 18

Clean classification CSV excludes other and uncertain rows. It has 7997 rows with only:

text,label

Heuristic rules implemented:

- Normalize raw text by lowercasing, removing punctuation, and collapsing whitespace.
- Direct aliases map common values like Dialog, Dialogue, First-person narrative, and Third-person narrator.
- Epistolary wins first when raw values mention letter, letters, diary, journal, correspondence, or memoir.
- mixed is used for explicit mixed/alternating modes, first+third combinations, or narrative+dialogue combinations.
- Strong dialogue markers map to dialogue: dialog, dialogue, conversation, characters speaking.
- Weak speech markers like Jesus speaking map to dialogue but are flagged uncertain.
- Missing or unmatched values map to other and are flagged uncertain.

Before/after examples:

Third-person narrative -> third_person
First-person narrative -> first_person
Dialog -> dialogue
Dialogue between two characters -> dialogue
Third-person narrative with dialogue -> mixed
First-person narrative (letters -> epistolary
Jesus speaking -> dialogue, uncertain
missing/blank -> other, uncertain

Verification:

python -m unittest discover -s tests

# Ran 13 tests: OK

Data validation:

classification_rows 7997
bad_rows 0
uncertain_count 229
