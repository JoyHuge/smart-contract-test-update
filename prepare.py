#!/usr/bin/env python3
"""Validate a user-provided project image before an LLM experiment."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.experiment_config import (
    DEFAULT_ENV_FILE,
    default_output_dir,
    load_dataset,
    preparation_report_path,
    read_env_file,
    resolve_setting,
    select_record,
)


ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "docker" / "compose.yml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one dataset sample, its project image, and the ground-truth test "
            "before calling an LLM."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--record-index", required=True, type=int)
    parser.add_argument(
        "--image",
        help="Docker image override. Default: SCT_DOCKER_IMAGE from .env.",
    )
    parser.add_argument(
        "--project-dir",
        help="Image project path override. Default: SCT_PROJECT_DIR from .env.",
    )
    parser.add_argument(
        "--framework",
        choices=("auto", "hardhat", "truffle", "foundry"),
        default="auto",
        help="Framework fallback when the dataset value is empty or unknown.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory. Default: results/<dataset-name>.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Environment file. Default: .env in the repository root.",
    )
    return parser.parse_args(argv)


def _compose_prefix(env_file: Path, project_name: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        project_name,
        "--env-file",
        str(env_file),
    ]


def _validate_docker(image: str) -> str:
    if shutil.which("docker") is None:
        raise ValueError("Docker is not installed or is not available on PATH.")
    if subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        raise ValueError("The Docker daemon is not available.")
    image_check = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_check.returncode != 0:
        raise ValueError(
            f"Docker image {image!r} is not installed. Build or pull it first."
        )
    image_id = image_check.stdout.strip()
    if not image_id:
        raise ValueError(f"Docker did not return an image ID for {image!r}.")
    return image_id


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        dataset = args.dataset.expanduser().resolve()
        records = load_dataset(dataset)
        record = select_record(records, args.record_index)
        env_file = args.env_file.expanduser().resolve()
        env_values = read_env_file(env_file)
        image = resolve_setting(args.image, "SCT_DOCKER_IMAGE", env_values)
        project_dir = resolve_setting(
            args.project_dir,
            "SCT_PROJECT_DIR",
            env_values,
            default="/workspace/project",
        )
        if not image:
            raise ValueError("Set SCT_DOCKER_IMAGE in .env or pass --image.")
        if not project_dir.startswith("/"):
            raise ValueError("SCT_PROJECT_DIR must be an absolute path inside the image.")
        image_id = _validate_docker(image)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else default_output_dir(dataset)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("Preparing sample")
    print(f"  Record: {args.record_index}")
    print(f"  Repository: {record.get('repo_name', '')}")
    print(f"  Commit: {record.get('test_commit_SHA', '')}")
    print(f"  Test: {record.get('test_file_path', '')}")
    print(f"  Image: {image}")

    run_id = hashlib.sha256(
        f"prepare\0{image}\0{dataset}\0{args.record_index}".encode("utf-8")
    ).hexdigest()[:12]
    compose = _compose_prefix(env_file, f"sct-prepare-{run_id}")
    report_path = preparation_report_path(output_dir, args.record_index)
    report_path.unlink(missing_ok=True)
    job_env = os.environ.copy()
    job_env.update(env_values)
    job_env.update(
        {
            "SCT_JOB_KIND": "prepare",
            "SCT_DOCKER_IMAGE": image,
            "SCT_DOCKER_IMAGE_ID": image_id,
            "SCT_AGENT_HOST_DIR": str(ROOT),
            "SCT_DATASET_HOST_PATH": str(dataset),
            "SCT_RESULT_HOST_DIR": str(output_dir),
            "SCT_PROJECT_DIR": project_dir,
            "SCT_RECORD_INDEX": str(args.record_index),
            "SCT_METHOD": "",
            "SCT_MODEL": "",
            "SCT_TEST_FRAMEWORK": "" if args.framework == "auto" else args.framework,
        }
    )

    result: subprocess.CompletedProcess | None = None
    try:
        result = subprocess.run(
            compose + ["run", "--rm", "sct-job"], env=job_env, check=False
        )
    finally:
        subprocess.run(
            compose + ["down", "--remove-orphans"],
            env=job_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    if result is None or result.returncode != 0:
        print(f"Preparation failed. See: {report_path}", file=sys.stderr)
        return 1
    if not report_path.is_file():
        print("Preparation did not produce its readiness report.", file=sys.stderr)
        return 1
    print(f"Sample is ready. Preparation report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
