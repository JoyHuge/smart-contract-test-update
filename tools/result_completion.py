"""Lightweight validation and indexing for completed SCT evaluations.

This module intentionally imports only the Python standard library so both the
local runner and the Docker matrix scheduler can use the exact same completion
rules before any heavy Agent dependencies or containers are started.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


JobKey = tuple[int, str, str]


def is_complete_evaluation(evaluation: dict) -> bool:
    if not isinstance(evaluation, dict):
        return False
    for key in (
        "record_index",
        "mode",
        "llm",
        "llm_run_result",
        "case_comparison",
        "ground_truth_test_result",
    ):
        if key not in evaluation:
            return False
    llm_run_result = evaluation.get("llm_run_result") or {}
    ground_truth_result = evaluation.get("ground_truth_test_result") or {}
    return (
        isinstance(llm_run_result, dict)
        and "compilation" in llm_run_result
        and isinstance(ground_truth_result, dict)
        and ground_truth_result.get("success") is True
    )


def evaluation_matches_record(evaluation: dict, config: dict, metadata: dict) -> bool:
    if evaluation.get("record_index") != config["record_index"]:
        return False

    eval_metadata = evaluation.get("metadata", {}) or {}
    if eval_metadata.get("repo_name", "") != metadata.get("repo_name", ""):
        return False

    eval_test_sha = eval_metadata.get("test_commit_SHA", "")
    record_test_sha = metadata.get("test_commit_SHA", "")
    if eval_test_sha and record_test_sha and eval_test_sha != record_test_sha:
        return False

    if eval_metadata.get("test_file_path", "") != metadata.get("test_file_path", ""):
        return False
    return True


def load_matching_evaluation(
    eval_path: Path,
    *,
    record_index: int,
    metadata: dict,
    mode: str,
    llm: str,
) -> dict | None:
    try:
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not is_complete_evaluation(evaluation):
        return None
    if evaluation.get("mode") != mode or evaluation.get("llm") != llm:
        return None
    config = {"record_index": int(record_index)}
    if not evaluation_matches_record(evaluation, config, metadata):
        return None
    return evaluation


def evaluation_job_key(evaluation: dict) -> JobKey | None:
    if not is_complete_evaluation(evaluation):
        return None
    try:
        return (
            int(evaluation["record_index"]),
            str(evaluation["mode"]),
            str(evaluation["llm"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


class CompletionIndex:
    """Index candidate eval.json files and validate metadata on lookup."""

    def __init__(self) -> None:
        self._paths: dict[JobKey, list[Path]] = defaultdict(list)

    def add_path(self, eval_path: Path) -> None:
        try:
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        key = evaluation_job_key(evaluation)
        if key is not None:
            self._paths[key].append(eval_path)

    def add_paths(self, eval_paths: Iterable[Path]) -> None:
        for eval_path in eval_paths:
            self.add_path(eval_path)

    def find(
        self,
        *,
        record_index: int,
        metadata: dict,
        mode: str,
        llm: str,
    ) -> Path | None:
        key = (int(record_index), mode, llm)
        for eval_path in self._paths.get(key, ()):
            if load_matching_evaluation(
                eval_path,
                record_index=record_index,
                metadata=metadata,
                mode=mode,
                llm=llm,
            ) is not None:
                return eval_path
        return None


def output_eval_paths(agent_dir: Path, methods: Iterable[str]) -> Iterable[Path]:
    """Yield only canonical output evals, excluding Docker result trees."""
    for method in methods:
        output_dir = agent_dir / f"output_{method}"
        if output_dir.is_dir():
            yield from output_dir.glob("*/*/*/eval.json")
