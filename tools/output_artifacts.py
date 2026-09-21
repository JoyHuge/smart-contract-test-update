"""Shared output-directory rules for local and Docker experiment runners."""
from __future__ import annotations

import json
from pathlib import Path


def _read_eval_record_index(eval_path: Path):
    try:
        with eval_path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("record_index")
    except Exception:
        return None


def resolve_output_test_dir(
    mode: str,
    project_name: str,
    test_file_stem: str,
    record_index: int,
    *,
    output_root: str | Path = ".",
) -> Path:
    """Return the per-record output directory without overwriting another record.

    The first record using a test stem gets ``<test_stem>``. Later records with
    the same stem get ``<test_stem>_2``, ``_3``, and so on. Existing output for
    the same record is reused so a rerun replaces only that record's latest
    model result.
    """
    project_output_dir = (
        Path(output_root) / f"output_{mode}" / project_name
    )
    suffix = 1

    while True:
        candidate_name = (
            test_file_stem if suffix == 1 else f"{test_file_stem}_{suffix}"
        )
        candidate = project_output_dir / candidate_name

        if not candidate.exists():
            return candidate

        eval_record_indices = {
            index
            for eval_path in candidate.glob("*/eval.json")
            if (index := _read_eval_record_index(eval_path)) is not None
        }
        if int(record_index) in eval_record_indices:
            return candidate
        if not eval_record_indices and not any(candidate.iterdir()):
            return candidate

        suffix += 1
