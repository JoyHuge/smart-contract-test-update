#!/usr/bin/env python3
"""Build dataset samples from projects selected by zero-based CSV indices."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dataset_builder.build_dataset_filelevel import build_for_repo, log_print
from dataset_builder.repo_scan import project_eligible


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECTS_FILE = PROJECT_ROOT / "docs" / "test-projects.csv"
DEFAULT_PROJECTS_DIR = PROJECT_ROOT / "projects"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone and process one or more projects selected by their "
            "zero-based row indices in test-projects.csv."
        )
    )
    parser.add_argument(
        "--project-indices",
        nargs="+",
        type=int,
        required=True,
        metavar="INDEX",
        help="One or more zero-based project indices, for example: 0 3 7.",
    )
    parser.add_argument(
        "--projects-file",
        type=Path,
        default=DEFAULT_PROJECTS_FILE,
        help=f"Project CSV file (default: {DEFAULT_PROJECTS_FILE}).",
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        default=DEFAULT_PROJECTS_DIR,
        help=f"Directory for cloned repositories (default: {DEFAULT_PROJECTS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSONL and log files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Exact JSONL output path. The log uses the same name with a .log "
            "suffix. Existing files are not overwritten."
        ),
    )
    return parser.parse_args()


def load_projects(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise ValueError(f"Project list not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"project_name", "github_url"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Project CSV must contain the columns: project_name, github_url"
            )

        projects = []
        for row_number, row in enumerate(reader, start=2):
            project_name = (row.get("project_name") or "").strip()
            github_url = (row.get("github_url") or "").strip()
            if not project_name or not github_url:
                raise ValueError(
                    f"Project CSV row {row_number} has an empty project_name or github_url"
                )
            projects.append(
                {"project_name": project_name, "github_url": github_url}
            )

    if not projects:
        raise ValueError(f"Project CSV contains no project records: {csv_path}")
    return projects


def validate_indices(indices: list[int], project_count: int) -> list[int]:
    invalid = [index for index in indices if index < 0 or index >= project_count]
    if invalid:
        valid_range = f"0-{project_count - 1}"
        values = ", ".join(str(index) for index in invalid)
        raise ValueError(
            f"Invalid project index or indices: {values}. Valid range: {valid_range}."
        )

    selected = []
    seen = set()
    for index in indices:
        if index not in seen:
            seen.add(index)
            selected.append(index)
    return selected


def safe_directory_name(project_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", project_name).strip("._")
    if not name:
        raise ValueError(f"Project name cannot be used as a directory: {project_name!r}")
    return name


def normalize_git_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:") :]
    elif url.startswith("ssh://git@github.com/"):
        url = "https://github.com/" + url[len("ssh://git@github.com/") :]
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


def run_git(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def validate_existing_repository(repo_dir: Path, expected_url: str) -> None:
    check = run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_dir)
    if check.returncode != 0 or check.stdout.strip() != "true":
        raise ValueError(
            f"Existing project directory is not a Git repository: {repo_dir}"
        )

    remote = run_git(["remote", "get-url", "origin"], cwd=repo_dir)
    if remote.returncode != 0:
        raise ValueError(f"Existing repository has no origin remote: {repo_dir}")
    if normalize_git_url(remote.stdout) != normalize_git_url(expected_url):
        raise ValueError(
            "Existing repository origin does not match the CSV entry: "
            f"{repo_dir} (origin={remote.stdout.strip()}, expected={expected_url})"
        )


def clone_or_reuse_project(
    project_name: str, github_url: str, projects_dir: Path
) -> tuple[Path, str]:
    repo_dir = projects_dir / safe_directory_name(project_name)
    if repo_dir.exists():
        if not repo_dir.is_dir():
            raise ValueError(
                f"Existing project path is not a directory: {repo_dir}"
            )
        validate_existing_repository(repo_dir, github_url)
        return repo_dir, "reused"

    projects_dir.mkdir(parents=True, exist_ok=True)
    temporary = projects_dir / f".clone-{repo_dir.name}-{uuid.uuid4().hex[:8]}"
    clone = run_git(["clone", "--", github_url, str(temporary)])
    if clone.returncode != 0:
        shutil.rmtree(temporary, ignore_errors=True)
        detail = clone.stderr.strip() or clone.stdout.strip() or "unknown Git error"
        raise RuntimeError(f"Failed to clone {github_url}: {detail}")

    try:
        temporary.replace(repo_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return repo_dir, "cloned"


def generate_output_paths(output_dir: Path) -> tuple[Path, Path]:
    date_text = datetime.now().strftime("%y%m%d")
    base = f"dataset_{date_text}"
    sequence = 1
    while True:
        suffix = f"{sequence:02d}"
        jsonl_path = output_dir / f"{base}_{suffix}.jsonl"
        log_path = output_dir / f"{base}_{suffix}.log"
        if not jsonl_path.exists() and not log_path.exists():
            return jsonl_path, log_path
        sequence += 1


def resolve_output_paths(
    output: Path | None, output_dir: Path
) -> tuple[Path, Path]:
    """Return explicit or automatically numbered dataset output paths."""
    if output is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return generate_output_paths(output_dir)

    jsonl_path = output.expanduser().resolve()
    if jsonl_path.suffix.lower() != ".jsonl":
        raise ValueError("--output must use the .jsonl file extension")
    log_path = jsonl_path.with_suffix(".log")
    existing = [path for path in (jsonl_path, log_path) if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise ValueError(f"Output file already exists: {paths}")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return jsonl_path, log_path


def print_sample_summary(jsonl_path: Path) -> None:
    """Print concise sample identifiers needed for image preparation."""
    print("\nGenerated samples:")
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            print(f"\n[{record_index}]")
            print(f"Repository: {record.get('repo_name', '')}")
            print(f"Framework: {record.get('test_framework', '') or 'auto'}")
            print(f"Commit: {record.get('test_commit_SHA', '')}")
            print(f"Contract: {record.get('production_file_path', '')}")
            print(f"Test: {record.get('test_file_path', '')}")
            print(f"Commit type: {record.get('commit_type', '')}")


def main() -> int:
    args = parse_args()

    if shutil.which("git") is None:
        print("Error: Git is required but was not found in PATH.", file=sys.stderr)
        return 2

    try:
        projects_file = args.projects_file.expanduser().resolve()
        projects_dir = args.projects_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        projects = load_projects(projects_file)
        indices = validate_indices(args.project_indices, len(projects))
        jsonl_path, log_path = resolve_output_paths(args.output, output_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    total_samples = 0
    failures = 0
    with jsonl_path.open("x", encoding="utf-8") as output_file, log_path.open(
        "x", encoding="utf-8"
    ) as log_file:
        log_print(f"Project list: {projects_file}", log_file)
        log_print(f"Selected indices: {', '.join(map(str, indices))}", log_file)
        log_print(f"Dataset file: {jsonl_path}", log_file)
        log_print(f"Log file: {log_path}", log_file)
        log_print("=" * 80, log_file)

        for position, index in enumerate(indices, start=1):
            project = projects[index]
            project_name = project["project_name"]
            github_url = project["github_url"]
            log_print(
                f"[{position}/{len(indices)}] index={index} project={project_name}",
                log_file,
            )
            try:
                repo_dir, repository_status = clone_or_reuse_project(
                    project_name, github_url, projects_dir
                )
                log_print(
                    f"  Repository: {repository_status} at {repo_dir}", log_file
                )
                if not project_eligible(repo_dir):
                    log_print(
                        "  Skipped: no eligible Solidity contract and test files found",
                        log_file,
                    )
                    continue

                count = build_for_repo(repo_dir, output_file, log_file)
                total_samples += count
                log_print(f"  Wrote {count} samples", log_file)
            except Exception as exc:
                failures += 1
                log_print(f"  ERROR: {exc}", log_file)

        log_print("=" * 80, log_file)
        log_print(f"Total samples: {total_samples}", log_file)
        log_print(f"Failed projects: {failures}", log_file)

    if total_samples == 0:
        jsonl_path.unlink(missing_ok=True)
        print(f"No samples were generated. See log: {log_path}")
    else:
        print(f"Dataset written to: {jsonl_path}")
        print(f"Log written to: {log_path}")
        print(f"Samples written: {total_samples}")
        print_sample_summary(jsonl_path)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
