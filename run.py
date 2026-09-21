#!/usr/bin/env python3
"""Run selected SCT experiments in Docker and aggregate results into Excel."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tools.excel_results import (
    aggregate_evaluations,
    apply_evaluation,
    apply_job_failure,
    ensure_result_workbook,
    find_evaluation,
)
from tools.experiment_config import (
    DEFAULT_ENV_FILE,
    default_output_dir,
    load_dataset,
    preparation_report_path,
    read_env_file,
    resolve_setting,
    select_record,
    validate_model_environment,
    validate_preparation_report,
)


ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "docker" / "compose.yml"

# Values accepted by --method. Multiple values may be comma-separated.
METHODS = (
    "SCT-Agent",
    "SCT-Agent-wo-TestRunner",
    "SCT-Agent-wo-AST-wo-TestRunner",
    "SDG",
    "SSR",
)

# Provider aliases accepted by --model. Multiple aliases may be comma-separated.
MODELS = ("deepseek", "glm", "gpt", "claude", "gemini", "qwen")


def _csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated value.")
    return list(dict.fromkeys(items))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one dataset record with selected methods and model providers, "
            "then update an Excel result workbook."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--record-index", required=True, type=int)
    parser.add_argument(
        "--method",
        type=_csv,
        required=True,
        help="Comma-separated experiment methods.",
    )
    parser.add_argument(
        "--model",
        type=_csv,
        required=True,
        help="Comma-separated model provider aliases.",
    )
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
        help="Host result directory. Default: results/<dataset-name>.",
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        help="Excel result path. Default: <output-dir>/test_results.xlsx.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Environment file. Default: .env in the repository root.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore completed results and run the requested jobs again.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Update Excel from existing eval.json files without using Docker or an LLM.",
    )
    parser.add_argument(
        "--skip-preparation-check",
        action="store_true",
        help="Advanced: run without requiring a matching prepare.py report.",
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


def _validate_selection(args: argparse.Namespace) -> None:
    unknown_methods = sorted(set(args.method) - set(METHODS))
    unknown_models = sorted(set(args.model) - set(MODELS))
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {', '.join(unknown_methods)}")
    if unknown_models:
        raise ValueError(f"Unknown model preset(s): {', '.join(unknown_models)}")


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
        _validate_selection(args)
        dataset = args.dataset.expanduser().resolve()
        records = load_dataset(dataset)
        record = select_record(records, args.record_index)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else default_output_dir(dataset)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_path = (
            args.excel_path.expanduser().resolve()
            if args.excel_path
            else output_dir / "test_results.xlsx"
        )
        ensure_result_workbook(excel_path, records, METHODS)

        if args.aggregate_only:
            updated = aggregate_evaluations(
                excel_path,
                output_dir,
                records,
                record_index=args.record_index,
                methods=args.method,
                models=args.model,
            )
            print(f"Excel updated from {updated} evaluation file(s): {excel_path}")
            return 0

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
        validate_model_environment(args.model, env_values)
        image_id = _validate_docker(image)
        if not args.skip_preparation_check:
            validate_preparation_report(
                preparation_report_path(output_dir, args.record_index),
                dataset=dataset,
                record_index=args.record_index,
                image=image,
                image_id=image_id,
                project_dir=project_dir,
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_id = hashlib.sha256(
        f"{image_id}\0{dataset}\0{args.record_index}".encode("utf-8")
    ).hexdigest()[:12]
    compose = _compose_prefix(env_file, f"sct-{run_id}")
    base_env = os.environ.copy()
    base_env.update(env_values)
    base_env.update(
        {
            "SCT_JOB_KIND": "experiment",
            "SCT_DOCKER_IMAGE": image,
            "SCT_DOCKER_IMAGE_ID": image_id,
            "SCT_AGENT_HOST_DIR": str(ROOT),
            "SCT_DATASET_HOST_PATH": str(dataset),
            "SCT_RESULT_HOST_DIR": str(output_dir),
            "SCT_PROJECT_DIR": project_dir,
            "SCT_RECORD_INDEX": str(args.record_index),
            "SCT_METHOD": args.method[0],
            "SCT_MODEL": args.model[0],
            "SCT_TEST_FRAMEWORK": "" if args.framework == "auto" else args.framework,
            "SCT_FORCE_RERUN": "1" if args.force_rerun else "0",
        }
    )

    print("Formal experiment")
    print(f"  Record: {args.record_index}")
    print(f"  Repository: {record.get('repo_name', '')}")
    print(f"  Methods: {', '.join(args.method)}")
    print(f"  Models: {', '.join(args.model)}")
    print(f"  Excel: {excel_path}")

    failures = 0
    try:
        for method in args.method:
            for model in args.model:
                job_env = dict(base_env)
                job_env.update({"SCT_METHOD": method, "SCT_MODEL": model})
                print(
                    f"[run] record={args.record_index} method={method} model={model}",
                    flush=True,
                )
                job_started_ns = time.time_ns()
                result = subprocess.run(
                    compose + ["run", "--rm", "sct-job"],
                    env=job_env,
                    check=False,
                )
                evaluation = None
                if result.returncode == 0:
                    evaluation = find_evaluation(
                        output_dir,
                        record,
                        record_index=args.record_index,
                        method=method,
                        model=model,
                        modified_after_ns=(
                            job_started_ns if args.force_rerun else None
                        ),
                    )

                if evaluation is None:
                    failures += 1
                    message = (
                        f"Docker job exited with code {result.returncode}"
                        if result.returncode != 0
                        else "Docker job produced no matching eval.json"
                    )
                    apply_job_failure(
                        excel_path,
                        records,
                        record_index=args.record_index,
                        method=method,
                        model=model,
                        message=message,
                    )
                    print(f"[failed] {method} / {model}: {message}", file=sys.stderr)
                    continue

                apply_evaluation(excel_path, records, evaluation)
                print(f"[saved] {method} / {model} -> {excel_path}")
    finally:
        subprocess.run(
            compose + ["down", "--remove-orphans"],
            env=base_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    print(f"Raw results: {output_dir}")
    print(f"Excel results: {excel_path}")
    if failures:
        print(f"{failures} job(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
