"""Parse Brahe-Novels analysis strings into structured records."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


LABEL_RE = re.compile(r"^\s*([^:\n]{1,80}):\s*(.*)$")

KNOWN_LABELS = {
    "Summary": "summary",
    "Enunciation": "enunciation_raw",
    "Tone": "tone_raw",
    "Genre": "genre_raw",
    "Speech standard": "speech_standard_raw",
    "Literary form": "literary_form_raw",
    "Active character": "active_character_raw",
    "Fuzzy time": "fuzzy_time_raw",
    "Fuzzy place": "fuzzy_place_raw",
}

OUTPUT_KEYS = [
    "summary",
    "enunciation_raw",
    "tone_raw",
    "genre_raw",
    "speech_standard_raw",
    "literary_form_raw",
    "active_character_raw",
    "fuzzy_time_raw",
    "fuzzy_place_raw",
]


def snake_case(label: str) -> str:
    """Convert a label such as 'Speech standard' to 'speech_standard'."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", label.strip().lower())
    return cleaned.strip("_")


def parse_analysis(analysis: str | None) -> dict[str, Any]:
    """Parse a multi-line ``Label: value`` analysis string."""
    parsed: dict[str, Any] = {key: None for key in OUTPUT_KEYS}
    other_labels: dict[str, str] = {}

    active_key: str | None = None
    active_is_other = False

    for line in (analysis or "").splitlines():
        match = LABEL_RE.match(line)
        if match:
            label, value = match.groups()
            label = label.strip()
            key = KNOWN_LABELS.get(label)
            if key is None:
                key = snake_case(label)
                other_labels[key] = value
                active_is_other = True
            else:
                parsed[key] = value
                active_is_other = False
            active_key = key
            continue

        if active_key is None:
            continue

        if active_is_other:
            existing = other_labels.get(active_key)
            other_labels[active_key] = _append_line(existing, line)
        else:
            parsed[active_key] = _append_line(parsed.get(active_key), line)

    parsed["other_labels"] = other_labels
    return parsed


def parse_record(row: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-like record with source fields and parsed analysis fields."""
    record = {
        "instruction_id": row.get("instruction_id"),
        "full_text": row.get("full_text"),
        "analysis": row.get("analysis"),
    }
    record.update(parse_analysis(row.get("analysis")))
    return record


def load_rows(dataset_path: str | Path) -> Iterable[dict[str, Any]]:
    """Load rows from a Hugging Face dataset saved on disk."""
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit(
            "The conversion step requires the optional 'datasets' package: "
            "python -m pip install datasets"
        ) from exc

    dataset = load_from_disk(str(dataset_path))
    split = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
    for row in split:
        yield dict(row)


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    count = 0
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(records: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    fieldnames = [
        "instruction_id",
        "full_text",
        "analysis",
        *OUTPUT_KEYS,
        "other_labels",
    ]
    count = 0
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["other_labels"] = json.dumps(row["other_labels"], ensure_ascii=False)
            writer.writerow(row)
            count += 1
    return count


def convert_dataset(dataset_path: str | Path, output_path: str | Path) -> int:
    records = (parse_record(row) for row in load_rows(dataset_path))
    if str(output_path).lower().endswith(".csv"):
        return write_csv(records, output_path)
    return write_jsonl(records, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/brahe_novels")
    parser.add_argument("--output", default="data/brahe_novels_parsed.jsonl")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    rows = list(load_rows(args.dataset_path))
    output_path = str(args.output).lower()
    writer = write_csv if output_path.endswith(".csv") else write_jsonl
    count = writer((parse_record(row) for row in rows), args.output)

    print(f"Wrote {count} records to {args.output}")
    for record in (parse_record(row) for row in rows[: args.examples]):
        print(json.dumps(record, ensure_ascii=False, indent=2))


def _append_line(existing: Any, line: str) -> str:
    if existing in (None, ""):
        return line
    return f"{existing}\n{line}"


if __name__ == "__main__":
    main()
