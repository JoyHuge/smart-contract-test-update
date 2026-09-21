"""Shared host-side configuration for preparation and experiment commands."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / ".env"


def read_env_file(path: Path) -> dict[str, str]:
    """Read non-empty values from a dotenv file."""
    if not path.is_file():
        raise ValueError(
            f"Environment file not found: {path}. Copy .env.example to .env first."
        )
    return {
        str(key): str(value)
        for key, value in dotenv_values(path).items()
        if key and value is not None and str(value).strip()
    }


def resolve_setting(
    cli_value: str | None,
    name: str,
    file_values: dict[str, str],
    *,
    default: str = "",
) -> str:
    """Resolve a setting with CLI, process environment, file, default priority."""
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip()
    process_value = os.environ.get(name, "").strip()
    if process_value:
        return process_value
    file_value = file_values.get(name, "").strip()
    return file_value or default


def load_dataset(path: Path) -> list[dict]:
    """Load and minimally validate a UTF-8 JSON Lines dataset."""
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}")

    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Dataset line {line_number} is not a JSON object"
                    )
                records.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON on dataset line {exc.lineno}: {exc.msg}"
        ) from exc

    if not records:
        raise ValueError(f"Dataset contains no records: {path}")
    return records


def select_record(records: list[dict], record_index: int) -> dict:
    """Return one zero-based dataset record or raise a clear error."""
    if record_index < 0 or record_index >= len(records):
        raise ValueError(
            f"Invalid record index {record_index}. Valid range: 0-{len(records) - 1}."
        )
    return records[record_index]


def default_output_dir(dataset: Path) -> Path:
    """Return the predictable result directory for a dataset."""
    return ROOT / "results" / dataset.stem


def dataset_digest(dataset: Path) -> str:
    """Return a stable digest used to bind preparation to dataset contents."""
    digest = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preparation_report_path(output_dir: Path, record_index: int) -> Path:
    return output_dir / "preparation" / f"record-{record_index:06d}.json"


def validate_preparation_report(
    report_path: Path,
    *,
    dataset: Path,
    record_index: int,
    image: str,
    image_id: str,
    project_dir: str,
) -> None:
    """Require a successful preparation report for the exact runtime inputs."""
    if not report_path.is_file():
        raise ValueError(
            "Preparation has not been completed for this sample. Run: "
            f"python prepare.py --dataset {dataset} --record-index {record_index}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid preparation report: {report_path}") from exc

    expected = {
        "dataset_sha256": dataset_digest(dataset),
        "record_index": int(record_index),
        "image": image,
        "image_id": image_id,
        "project_dir": project_dir,
    }
    mismatches = [
        key for key, value in expected.items() if report.get(key) != value
    ]
    if report.get("success") is not True or mismatches:
        detail = f" Mismatched fields: {', '.join(mismatches)}." if mismatches else ""
        raise ValueError(
            "Preparation is missing, failed, or no longer matches this run."
            f"{detail} Run prepare.py again."
        )


def validate_model_environment(
    models: list[str], file_values: dict[str, str]
) -> None:
    """Require the three provider settings used by every selected model."""
    missing: list[str] = []
    for model in models:
        prefix = model.upper()
        for suffix in ("API_KEY", "BASE_URL", "MODEL"):
            name = f"{prefix}_{suffix}"
            if not (os.environ.get(name, "").strip() or file_values.get(name, "").strip()):
                missing.append(name)
    if missing:
        raise ValueError(
            "Missing model configuration in the environment file: "
            + ", ".join(missing)
        )
