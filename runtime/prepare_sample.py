#!/usr/bin/env python3
"""Validate one dataset sample and its ground-truth test inside the project image."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from base import (
    detect_framework,
    ground_truth_test_cache_key,
    init_components,
    save_ground_truth_test_cache,
)
from tools.dataset_loader import DatasetLoader


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_source(value: str) -> str:
    return (value or "").replace("\r\n", "\n").rstrip("\n")


def _git_head(project_dir: Path) -> str:
    if not (project_dir / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    dataset_path = Path(os.environ["SCT_DATASET_PATH"])
    project_dir = Path(os.environ["SCT_PROJECT_DIR"]).resolve()
    result_dir = Path(os.environ.get("SCT_RESULT_DIR", "/results"))
    record_index = int(os.environ["SCT_RECORD_INDEX"])
    image = os.environ.get("SCT_DOCKER_IMAGE", "")
    framework_override = os.environ.get("SCT_TEST_FRAMEWORK", "")
    report_path = result_dir / "preparation" / f"record-{record_index:06d}.json"

    report = {
        "success": False,
        "dataset_sha256": _dataset_sha256(dataset_path),
        "record_index": record_index,
        "image": image,
        "image_id": os.environ.get("SCT_DOCKER_IMAGE_ID", ""),
        "project_dir": str(project_dir),
        "framework": "",
        "expected_commit": "",
        "image_commit": "",
        "commit_verified": False,
        "test_source_verified": False,
        "ground_truth_test": {},
        "error": "",
    }

    try:
        dataset = DatasetLoader(str(dataset_path))
        if dataset.load_dataset() == 0:
            raise RuntimeError("The dataset contains no records")
        if not dataset.set_current_record(record_index):
            raise RuntimeError(f"Invalid record index: {record_index}")

        validation = dataset.validate_record()
        if validation.get("errors"):
            raise RuntimeError("; ".join(validation["errors"]))

        metadata = dataset.get_metadata()
        report["expected_commit"] = metadata.get("test_commit_SHA", "")
        config = {
            "project_dir": str(project_dir),
            "dataset_path": str(dataset_path),
            "record_index": record_index,
            "test_framework": framework_override,
            "ground_truth_test_cache_path": os.environ.get(
                "SCT_GROUND_TRUTH_TEST_CACHE_PATH", ""
            ),
        }
        components = init_components(config)
        metadata_framework = str(
            metadata.get("test_framework", "") or ""
        ).strip().lower()
        if metadata_framework in {"", "unknown", "auto", "none", "n/a", "na"}:
            framework = detect_framework(config, components)
        else:
            framework = metadata_framework
        report["framework"] = framework

        image_commit = _git_head(project_dir)
        report["image_commit"] = image_commit
        if image_commit and report["expected_commit"]:
            report["commit_verified"] = image_commit == report["expected_commit"]
            if not report["commit_verified"]:
                raise RuntimeError(
                    "The project image is at a different Git commit "
                    f"(image={image_commit}, expected={report['expected_commit']})"
                )

        original_test_path = metadata.get("test_file_path", "")
        ground_truth_source = components["file_manager"].get_original_test(
            original_test_path
        )
        if ground_truth_source is None:
            raise RuntimeError(
                f"Ground-truth test not found in the image: {original_test_path}"
            )
        report["test_source_verified"] = _normalized_source(
            ground_truth_source
        ) == _normalized_source(dataset.get_test_file_after())
        if not report["test_source_verified"]:
            raise RuntimeError(
                "The ground-truth test in the image does not match test_file_after "
                "in the selected dataset record"
            )

        ground_truth_abs_path = components["file_manager"]._normalize_path(
            original_test_path
        )
        try:
            ground_truth_relative_path = str(
                ground_truth_abs_path.relative_to(project_dir)
            )
        except ValueError:
            ground_truth_relative_path = os.path.relpath(
                ground_truth_abs_path, project_dir
            )
        success, log = components["runner"].run(
            ground_truth_relative_path, framework=framework
        )
        run_result = components["comparator"].parse_test_run_result(
            log, framework=framework
        )
        passing_cases = components["comparator"].parse_passing_tests_from_log(log)
        report["ground_truth_test"] = {
            "success": success,
            "compilation": run_result.get("compilation"),
            "test_result": run_result.get("test_result"),
            "passing_count": len(passing_cases),
            "parsed_passing_count": run_result.get("passing_count", 0),
            "failing_count": run_result.get("failing_count", 0),
            "passing_cases": passing_cases,
            "log": log,
        }
        if not success:
            raise RuntimeError(
                "The ground-truth test did not pass in the supplied image"
            )

        cache_entry = {
            "record_index": record_index,
            "repo_name": metadata.get("repo_name", ""),
            "test_commit_SHA": metadata.get("test_commit_SHA", ""),
            "test_file_path": metadata.get("test_file_path", ""),
            "framework": framework,
            **report["ground_truth_test"],
        }
        save_ground_truth_test_cache(
            config,
            ground_truth_test_cache_key(config, metadata, framework),
            cache_entry,
        )
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    _write_json_atomic(report_path, report)
    if report["success"]:
        ground_truth = report["ground_truth_test"]
        print("Preparation passed.")
        print(f"Framework: {report['framework']}")
        print(
            "Ground-truth test: PASS "
            f"({ground_truth.get('passing_count', 0)} passing)"
        )
        print(f"Report: {report_path}")
        return 0

    print("Preparation failed.", file=sys.stderr)
    print(report["error"], file=sys.stderr)
    print(f"Report: {report_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
