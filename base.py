"""Shared infrastructure for the SDG, SSR, and SCT-Agent methods."""

import json
import os
import re
import hashlib
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI

from tools.dataset_loader import DatasetLoader
from tools.output_artifacts import resolve_output_test_dir
from tools.test_file_manager import TestFileManager
from tools.test_comparator import TestComparator
from tools.framework_detector import FrameworkDetector
from tools.contract_project_context import ContractProjectContext
from tools.result_completion import (
    evaluation_matches_record,
    is_complete_evaluation as _is_complete_evaluation,
    load_matching_evaluation,
)
from runtime.test_runner import TestRunner


_OUTPUT_ARCHIVE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

SUPPORTED_LLM_NAMES = (
    "deepseek",
    "glm",
    "gpt",
    "claude",
    "gemini",
    "qwen",
)


def _build_llm_timeout(config: dict) -> httpx.Timeout:
    """Keep long model reads while failing broken connections promptly."""
    timeout_values = {
        "connect": float(config.get("connect_timeout", 30)),
        "read": float(config.get("timeout", 1200)),
        "write": float(config.get("write_timeout", 60)),
        "pool": float(config.get("pool_timeout", 60)),
    }
    invalid = {name: value for name, value in timeout_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"LLM timeout values must be positive: {invalid}")
    return httpx.Timeout(**timeout_values)


def resolve_model_list(llm_setting) -> list[str]:
    """Resolve config['LLM'] into supported provider aliases.

    Recommended config forms:
      - "LLM": ["glm"] for a single model
      - "LLM": ["glm", "deepseek"] for selected models
    """
    if not isinstance(llm_setting, list):
        raise ValueError("config['LLM'] must be a list, e.g. ['glm'] or ['glm', 'deepseek'].")

    model_list = [str(part).strip() for part in llm_setting if str(part).strip()]
    if not model_list:
        raise ValueError("config['LLM'] must contain at least one model preset.")

    unknown = [name for name in model_list if name not in SUPPORTED_LLM_NAMES]
    if unknown:
        known = ", ".join(SUPPORTED_LLM_NAMES)
        raise ValueError(
            f"Unknown LLM provider alias(es): {unknown}. "
            f"Use a list like ['glm'] or ['glm', 'deepseek']. "
            f"Known presets: {known}."
        )
    return model_list


def load_llm_environment(name: str) -> dict[str, str]:
    """Load one model provider entirely from environment variables."""
    if name not in SUPPORTED_LLM_NAMES:
        known = ", ".join(SUPPORTED_LLM_NAMES)
        raise ValueError(f"Unknown LLM provider alias: {name!r}. Use one of: {known}")

    prefix = name.upper()
    variable_names = {
        "api_key": f"{prefix}_API_KEY",
        "base_url": f"{prefix}_BASE_URL",
        "model": f"{prefix}_MODEL",
    }
    settings = {
        key: os.environ.get(variable_name, "").strip()
        for key, variable_name in variable_names.items()
    }
    missing = [
        variable_name
        for key, variable_name in variable_names.items()
        if not settings[key]
    ]
    if missing:
        raise ValueError(
            f"LLM={name!r} requires these variables in .env: {', '.join(missing)}."
        )
    return settings


def create_chat_llm(config: dict) -> ChatOpenAI:
    """Build a ChatOpenAI client using only environment configuration."""
    name = str(config.get("LLM", "")).strip()
    settings = load_llm_environment(name)

    request_timeout = _build_llm_timeout(config)
    kwargs_llm: dict = {
        "api_key": settings["api_key"],
        "base_url": settings["base_url"],
        "model": settings["model"],
        "temperature": config["temperature"],
        "timeout": request_timeout,
        "max_retries": int(config.get("max_retries", 0)),
    }
    if config.get("streaming", False):
        kwargs_llm["streaming"] = True
        # Preserve token accounting when invoke() aggregates SSE chunks.
        kwargs_llm["stream_usage"] = bool(config.get("stream_usage", True))
    mout = config.get("max_output_tokens", 0)
    if mout and int(mout) > 0:
        kwargs_llm["max_tokens"] = int(mout)
    return ChatOpenAI(**kwargs_llm)


def is_retryable_llm_error(exc: Exception) -> bool:
    retryable_status_codes = {500, 502, 503, 504}
    retryable_names = {
        "APITimeoutError",
        "APIConnectionError",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "TimeoutException",
    }
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in retryable_names:
            return True
        try:
            status_code = int(getattr(current, "status_code", 0) or 0)
        except (TypeError, ValueError):
            status_code = 0
        if status_code in retryable_status_codes:
            return True
        current = current.__cause__
    return False


def invoke_llm_with_retry(llm, messages, config: dict | None = None):
    """Invoke LLM with retries for transient network, timeout, and 5xx failures."""
    config = config or {}
    max_attempts = max(1, int(config.get("llm_invoke_retries", 3)))
    retry_wait = max(1, int(config.get("llm_retry_wait", 30)))

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            if not is_retryable_llm_error(exc) or attempt >= max_attempts:
                raise
            root = exc
            seen: set[int] = set()
            while getattr(root, "__cause__", None) is not None and id(root) not in seen:
                seen.add(id(root))
                root = root.__cause__
            detail = f"; cause={type(root).__name__}: {root}" if root is not exc else ""
            wait_seconds = min(retry_wait * attempt, 120)
            print(
                f"  LLM call failed ({type(exc).__name__}); "
                f"retry {attempt}/{max_attempts} in {wait_seconds}s...{detail}"
            )
            time.sleep(wait_seconds)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM invoke failed without an exception.")


def should_skip_completed_run(config: dict) -> bool:
    """Return True when completed (mode, model) runs should be skipped.

    Priority: force_rerun > skip_completed config (default True).
    Set config['skip_completed'] = False in main_*.py for a full rerun in PyCharm.
    """
    if config.get("force_rerun"):
        return False
    return bool(config.get("skip_completed", True))


def infer_project_name_from_metadata(metadata: dict) -> str:
    for key in ("production_file_path", "test_file_path"):
        path = metadata.get(key, "")
        parts = Path(path).parts
        if parts:
            return parts[0]

    repo_name = metadata.get("repo_name", "")
    return repo_name


def get_completed_eval_path(config: dict, metadata: dict, mode: str, llm: str) -> Path | None:
    project_name = infer_project_name_from_metadata(metadata)
    test_file_stem = Path(metadata["test_file_path"]).stem
    output_test_dir = resolve_output_test_dir(
        mode, project_name, test_file_stem, config["record_index"]
    )
    eval_path = output_test_dir / llm / "eval.json"
    if not eval_path.exists():
        return None

    if load_matching_evaluation(
        eval_path,
        record_index=config["record_index"],
        metadata=metadata,
        mode=mode,
        llm=llm,
    ) is None:
        return None
    return eval_path


def is_run_completed(config: dict, metadata: dict, mode: str, llm: str) -> bool:
    return get_completed_eval_path(config, metadata, mode, llm) is not None


def print_skip_completed_banner(config: dict) -> None:
    if should_skip_completed_run(config):
        print("\nSkip completed runs: ON (set config['skip_completed'] = False for full rerun)")
    else:
        print("\nSkip completed runs: OFF (full rerun)")


def parallel_models_enabled(config: dict) -> bool:
    return bool(config.get("parallel_models")) or _env_flag("SCT_PARALLEL_MODELS")


def run_model_jobs(config: dict, model_list: list[str], job_fn, label: str) -> None:
    """Run one job per configured model, optionally in parallel.

    Parallelism is intentionally model-scoped: the worker count is the number of
    configured models, not a separately tuned number.
    """
    if not parallel_models_enabled(config) or len(model_list) <= 1:
        for model_name in model_list:
            job_fn(model_name)
        return

    max_workers = len(model_list)
    print(f"\n[parallel] {label}: running {max_workers} configured model(s) in parallel")
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(job_fn, model_name): model_name for model_name in model_list
        }
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((model_name, exc))
                print(f"\n[parallel] {label} / {model_name} failed: {type(exc).__name__}: {exc}")
    if failures:
        failed_models = ", ".join(model for model, _ in failures)
        raise RuntimeError(f"{label} failed for model(s): {failed_models}")


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a generated artifact atomically, leaving no partial destination."""
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict, destination: Path) -> None:
    """Write JSON atomically so an interrupted run cannot leave a false archive."""
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def archive_generated_test_source(
    source_path: Path,
    destination_dir: Path,
    metadata: dict,
    mode: str,
    model_name: str,
    original_test_file: str,
    record_index: int,
) -> dict:
    """Persist one generated test before the outer experiment can be interrupted.

    The source is removed only after both the source copy and its metadata have
    been atomically committed to the per-model output directory.  Returning a
    status instead of raising keeps evaluation results available even when an
    archival filesystem error occurs; the source remains in place for the
    existing end-of-script recovery sweep in run.py.
    """
    result = {
        "archived": False,
        "archive_path": "",
        "archive_error": "",
    }
    if not source_path.is_file():
        result["archive_error"] = f"Generated test source is missing: {source_path}"
        return result

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name
    metadata_path = destination_dir / "generated_test_metadata.json"
    archive_metadata = {
        "record_index": record_index,
        "repo_name": metadata.get("repo_name", ""),
        "test_commit_SHA": metadata.get("test_commit_SHA", ""),
        "mode": mode,
        "model": model_name,
        "original_test_file": original_test_file,
        "archived_from": str(source_path),
        "archived_to": str(destination),
    }

    try:
        _atomic_copy_file(source_path, destination)
        _atomic_write_json(archive_metadata, metadata_path)
        source_path.unlink()
    except Exception as exc:
        result["archive_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["archived"] = True
    result["archive_path"] = str(destination)
    return result


def sources_cache_path(
    config: dict,
    metadata: dict,
    info: list[int],
    cache_name: str,
) -> Path | None:
    configured = config.get("sources_cache_dir") or os.environ.get("SCT_SOURCES_CACHE_DIR")
    if not configured:
        return None

    payload = {
        "cache_name": cache_name,
        "record_index": int(config["record_index"]),
        "repo_name": metadata.get("repo_name", ""),
        "test_commit_SHA": metadata.get("test_commit_SHA", ""),
        "production_file_path": metadata.get("production_file_path", ""),
        "test_file_path": metadata.get("test_file_path", ""),
        "info": sorted(int(item) for item in info),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return Path(configured) / f"{cache_name}_{digest}.json"


def load_cached_sources(
    config: dict,
    metadata: dict,
    info: list[int],
    cache_name: str,
) -> dict | None:
    cache_path = sources_cache_path(config, metadata, info, cache_name)
    if cache_path is None or not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    sources = data.get("sources")
    if not isinstance(sources, dict):
        return None

    print(f"  Source cache hit: {cache_path}")
    return sources


def save_cached_sources(
    config: dict,
    metadata: dict,
    info: list[int],
    cache_name: str,
    sources: dict,
) -> None:
    cache_path = sources_cache_path(config, metadata, info, cache_name)
    if cache_path is None:
        return

    serializable_sources = {
        key: value
        for key, value in sources.items()
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({"sources": serializable_sources}, f, indent=2, ensure_ascii=False)


def ground_truth_test_cache_path(config: dict) -> Path:
    configured = config.get("ground_truth_test_cache_path") or os.environ.get(
        "SCT_GROUND_TRUTH_TEST_CACHE_PATH"
    )
    if configured:
        return Path(configured)

    base_path = Path(config.get("dataset_path") or ".").resolve()
    base_dir = base_path.parent if base_path.suffix else base_path
    return base_dir / ".cache" / "ground_truth_test_results.json"


def ground_truth_test_cache_key(config: dict, metadata: dict, framework: str) -> str:
    payload = {
        "record_index": int(config["record_index"]),
        "repo_name": metadata.get("repo_name", ""),
        "test_commit_SHA": metadata.get("test_commit_SHA", ""),
        "test_file_path": metadata.get("test_file_path", ""),
        "framework": framework,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def load_ground_truth_test_cache(config: dict, cache_key: str) -> dict | None:
    cache_path = ground_truth_test_cache_path(config)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return None

    entry = cache.get(cache_key)
    if not isinstance(entry, dict) or entry.get("success") is not True:
        return None
    if not isinstance(entry.get("passing_cases"), list) and not entry.get("log"):
        return None
    return entry


def save_ground_truth_test_cache(config: dict, cache_key: str, entry: dict) -> None:
    if entry.get("success") is not True:
        return

    cache_path = ground_truth_test_cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    cache[cache_key] = entry
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def count_top_level_truffle_contracts(source: str) -> int:
    if not source or not source.strip():
        return 0
    return len(re.findall(r"\bcontract\s*\(\s*['\"]", source))


def extract_code_from_llm(text: str) -> str:
    """Extract test code while preferring JavaScript or TypeScript fences.

    Explanatory Solidity and JSON blocks must not interfere with extraction.
    """
    if not text or not text.strip():
        return ""

    fence_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    blocks = []
    for match in fence_pattern.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        code = match.group(2).strip()
        if code:
            blocks.append((lang, code))

    if blocks:
        js_langs = {"javascript", "js", "typescript", "ts"}
        js_chunks = [code for lang, code in blocks if lang in js_langs]
        if js_chunks:
            return "\n\n".join(js_chunks)

        test_like_chunks = [
            code for _, code in blocks
            if (
                "artifacts.require" in code
                or re.search(r"\b(?:contract|describe|it)\s*\(", code)
                or "expect(" in code
                or "assert" in code
            )
        ]
        if test_like_chunks:
            return "\n\n".join(test_like_chunks)

        return blocks[-1][1]

    return text.strip()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def validate_test_syntax_impl(test_content: str) -> dict:
    errors = []

    artifacts_contracts = re.findall(
        r'artifacts\.require\([\'"]([^\'"]+)[\'"]\)', test_content
    )
    for contract in artifacts_contracts:
        name = contract.split('/')[-1].replace('.sol', '')
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            errors.append(f"Invalid contract name format: {contract}")

    has_contract = bool(re.search(r'contract\s*\(\s*[\'"]', test_content))
    has_describe = bool(re.search(r'describe\s*\(\s*[\'"]', test_content))
    if not has_contract and not has_describe:
        errors.append("No contract/describe definition found")

    if 'async function' in test_content and 'await' not in test_content:
        errors.append("Test uses async function but no await found")

    if test_content.count('{') != test_content.count('}'):
        errors.append("Unbalanced curly braces")
    if test_content.count('(') != test_content.count(')'):
        errors.append("Unbalanced parentheses")

    if not artifacts_contracts:
        used_contracts = set(re.findall(
            r'(?:var|let|const)\s+\w+\s*=\s*await\s+([A-Z][a-zA-Z0-9]*)\.new\(',
            test_content
        ))
        used_contracts.update(re.findall(
            r'(?:^|\n)\s*\w+\s*=\s*await\s+([A-Z][a-zA-Z0-9]*)\.new\(',
            test_content
        ))
        used_contracts.update(re.findall(
            r'=\s*([A-Z][a-zA-Z0-9]*)\.at\(', test_content
        ))
        used_contracts.update(re.findall(
            r'=\s*await\s+([A-Z][a-zA-Z0-9]*)\.deployed\(', test_content
        ))
        non_contract = {'Promise', 'BigNumber', 'Web3', 'Assert', 'Expect'}
        used_contracts -= non_contract
        if used_contracts:
            errors.append(
                f"Contracts used but no artifacts.require found: {', '.join(sorted(used_contracts))}"
            )

    return {
        'valid': len(errors) == 0,
        'errors': errors if errors else ["No syntax errors found"]
    }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

INFO_NAMES = {
    1: "contract_file_before",
    2: "contract_file_after",
    3: "contract_diff",
    4: "test_file_before",
    5: "contract_context",
    6: "error_log",
    7: "all_project_contracts",
    8: "project_config",
}


def build_section(info_id: int, header: str, content: str, enabled: list[int]) -> str:
    if info_id not in enabled or not content:
        return ""
    return f"\n=== {header} ===\n{content}\n"


def build_generate_prompt(info: list[int], **kwargs) -> str:
    """Build the generation prompt shared by all three methods."""
    prompt = """You are a smart-contract test evolution expert.

TASK: The smart contract code has been modified. Based on the information provided below, update the ORIGINAL TEST FILE so that it works correctly with the CURRENT version of the smart contracts.

Read through all the information to understand the project context. Pay special attention to:
- CONTRACT DIFF: shows exactly what changed in the contract.
- CONTRACT FILE BEFORE / AFTER: the full contract code before and after the change.
- ORIGINAL TEST FILE: the test file you need to update.

Based on your understanding of the contract changes, modify the ORIGINAL TEST FILE accordingly. If the changes do not affect the test's behavior, return the test file unchanged.

"""
    sections = [
        (1, "CONTRACT FILE BEFORE CHANGE (complete file, OLD version)", kwargs.get('contract_file_before', '')),
        (2, "CONTRACT FILE AFTER CHANGE (complete file, NEW version)", kwargs.get('contract_file_after', '')),
        (3, "CONTRACT DIFF (git diff showing what changed)", kwargs.get('contract_diff', '')),
        (4, "COMPLETE ORIGINAL TEST FILE (before change, modify this file)", kwargs.get('test_file_before', '')),
        (5, "CURRENT CONTRACT CODE (referenced by test)", kwargs.get('contract_context', '')),
        (7, "ALL PROJECT CONTRACT CODE (authoritative API context)", kwargs.get('all_project_contracts', '')),
        (8, "PROJECT CONFIG AND UTILITIES (config, helpers, migrations)", kwargs.get('project_config', '')),
    ]
    for info_id, header, content in sections:
        prompt += build_section(info_id, header, content, info)

    prompt += """
OUTPUT:
- Return the COMPLETE updated test file wrapped in ```javascript ... ```.
- Preserve the original file structure, all assertion styles, helper methods, imports, and test setup.
- Only modify the parts that must change to work with the updated contract.
- Your output MUST contain the ENTIRE file. Do NOT truncate or stop mid-file. The output should have roughly the same length as the ORIGINAL TEST FILE.
"""
    return prompt


def trim_test_log_for_fix(log: str, max_lines: int = 400) -> str:
    if not (log and log.strip()):
        return ""
    lines = log.strip().split("\n")
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def build_fix_prompt(info: list[int], test_content: str, **kwargs) -> str:
    """Build the repair prompt used by SCT-Agent."""
    prompt = """You are a smart-contract test debugger.

The test file below FAILED when executed. Fix ALL errors based on the error message and any contract code provided below.

=== FAILED TEST FILE ===
"""
    prompt += test_content + "\n"

    sections = [
        (5, "CURRENT CONTRACT CODE (the actual API your test must work with)", kwargs.get("contract_context", "")),
        (7, "ALL PROJECT CONTRACT CODE (authoritative API and behavior context)", kwargs.get("all_project_contracts", "")),
        (8, "PROJECT CONFIG AND UTILITIES (config, helpers, migrations)", kwargs.get("project_config", "")),
        (6, "FULL TEST OUTPUT", kwargs.get("error_log", "")),
    ]
    for info_id, header, content in sections:
        prompt += build_section(info_id, header, content, info)

    prompt += """
INSTRUCTIONS:
- Fix ALL errors shown in FULL TEST OUTPUT; there may be multiple failing tests. Address every failure.
- Do NOT blindly change assertion expected values to match runtime values. Use the contract code to determine the correct outcome first.
- Output the COMPLETE fixed test file, wrapped in ```javascript ... ``` markers."""
    return prompt


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def get_all_project_contracts(project_context: ContractProjectContext) -> str:
    all_paths = project_context.get_all_contracts()
    parts = []
    for path in all_paths:
        contract = project_context.load_contract(path)
        if contract:
            parts.append(f"--- {path} ---\n{contract.content}")
    return "\n\n".join(parts)


def get_project_config(project_dir: str) -> str:
    project_path = Path(project_dir)
    parts = []

    config_files = [
        "truffle-config.js", "truffle.js",
        "hardhat.config.js", "hardhat.config.ts",
        "package.json",
    ]

    helper_patterns = [
        "test/helper.js", "test/helpers.js",
        "test/test-helper.js", "test/test-helpers.js",
        "test/test-setup.js",
    ]

    import glob
    migration_files = sorted(glob.glob(str(project_path / "migrations" / "*.js")))

    for filename in config_files:
        filepath = project_path / filename
        if filepath.exists():
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
                parts.append(f"--- {filename} ---\n{content}")
            except Exception:
                pass

    for rel_path in helper_patterns:
        filepath = project_path / rel_path
        if filepath.exists():
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
                parts.append(f"--- {rel_path} ---\n{content}")
            except Exception:
                pass

    for mig_path in migration_files:
        filepath = Path(mig_path)
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            rel = filepath.relative_to(project_path).as_posix()
            parts.append(f"--- {rel} ---\n{content}")
        except Exception:
            pass

    return "\n\n".join(parts)


def print_info_status(info: list[int]):
    print("\nEnabled info sources:")
    for i, name in INFO_NAMES.items():
        status = "ON" if i in info else "OFF"
        print(f"  {i}: {name:30s} {status}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def collect_info_sources(config: dict, dataset: DatasetLoader, info: list[int]) -> dict:
    """Collect enabled information sources.

    Source 6, ``error_log``, is managed by each method and is not collected here.
    """
    metadata = dataset.get_metadata()
    cached_sources = load_cached_sources(config, metadata, info, cache_name="info")
    if cached_sources is not None:
        cached_sources["project_context"] = ContractProjectContext(config['project_dir'])
        for key in (
            "contract_file_before",
            "contract_file_after",
            "contract_diff",
            "test_file_before",
            "contract_context",
            "all_project_contracts",
            "project_config",
        ):
            print(f"  [cache] {key}: {len(str(cached_sources.get(key, '')))} chars")
        return cached_sources

    project_context = ContractProjectContext(config['project_dir'])

    contract_file_before = dataset.get_contract_file_before() if 1 in info else ""
    print(f"  [1] contract_file_before: {len(contract_file_before)} chars")

    contract_file_after = dataset.get_contract_file_after() if 2 in info else ""
    print(f"  [2] contract_file_after:  {len(contract_file_after)} chars")

    contract_diff = dataset.get_contract_diff() if 3 in info else ""
    print(f"  [3] contract_diff:        {len(contract_diff)} chars")

    test_file_before = dataset.get_test_file_before() if 4 in info else ""
    print(f"  [4] test_file_before:     {len(test_file_before)} chars")

    if 5 in info:
        print("  Fetching contract context (referenced by test)...")
        required_contracts = re.findall(
            r'artifacts\.require\(["\']([^"\']+)["\']\)', test_file_before or ""
        )
        if required_contracts:
            contract_context = project_context.get_contracts_by_names(
                required_contracts, max_context_length=12000,
            )
        else:
            contract_context = project_context.get_context_for_test(
                metadata['production_file_path'],
                max_context_length=12000,
            )
        print(f"  [5] contract_context:     {len(contract_context)} chars ({len(required_contracts)} contracts)")
    else:
        contract_context = ""

    if 7 in info:
        print("  Fetching all project contract code...")
        all_project_contracts = get_all_project_contracts(project_context)
        print(f"  [7] all_project_contracts: {len(all_project_contracts)} chars")
    else:
        all_project_contracts = ""

    if 8 in info:
        print("  Fetching project config and utilities...")
        project_config = get_project_config(config['project_dir'])
        print(f"  [8] project_config:      {len(project_config)} chars")
    else:
        project_config = ""

    sources = {
        'contract_file_before': contract_file_before,
        'contract_file_after': contract_file_after,
        'contract_diff': contract_diff,
        'test_file_before': test_file_before,
        'contract_context': contract_context,
        'all_project_contracts': all_project_contracts,
        'project_config': project_config,
        'project_context': project_context,
    }
    save_cached_sources(config, metadata, info, cache_name="info", sources=sources)
    return sources


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _resolve_dataset_project_prefix(config: dict) -> str:
    prefix = str(config.get("dataset_project_prefix") or "").strip()
    if prefix:
        return prefix
    if not config.get("dataset_path") or config.get("record_index") is None:
        return ""
    from tools.dataset_loader import DatasetLoader

    dataset = DatasetLoader(config["dataset_path"])
    if dataset.load_dataset() == 0:
        return ""
    if not dataset.set_current_record(config["record_index"]):
        return ""
    return infer_project_name_from_metadata(dataset.get_metadata())


def init_components(config: dict) -> dict:
    """Initialize and return all runtime components."""
    dataset_project_prefix = _resolve_dataset_project_prefix(config)
    return {
        'file_manager': TestFileManager(
            config['project_dir'],
            dataset_project_prefix=dataset_project_prefix,
        ),
        'comparator': TestComparator(),
        'runner': TestRunner(config['project_dir'], record_index=config.get('record_index')),
        'framework_detector': FrameworkDetector(config['project_dir']),
    }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def generate_test(llm, config: dict, info: list[int], sources: dict) -> str:
    """Invoke the LLM once and return a complete generated test file."""
    generate_prompt = build_generate_prompt(
        info,
        contract_file_before=sources['contract_file_before'],
        contract_file_after=sources['contract_file_after'],
        contract_diff=sources['contract_diff'],
        test_file_before=sources['test_file_before'],
        contract_context=sources['contract_context'],
        all_project_contracts=sources['all_project_contracts'],
        project_config=sources['project_config'],
    )
    print(f"  Prompt length: {len(generate_prompt)} chars")

    try:
        response = invoke_llm_with_retry(llm, [("user", generate_prompt)], config)
        test_code = extract_code_from_llm(response.content)
        print(f"  Generated: {len(test_code)} chars")
        return test_code
    except Exception as e:
        print(f"  Generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def detect_framework(config: dict, components: dict) -> str:
    framework = str(config.get('test_framework', '') or '').strip().lower()
    if framework in {"unknown", "auto", "none", "n/a", "na"}:
        framework = ""
    if not framework:
        framework = components['framework_detector'].detect()
        if not framework:
            framework = "hardhat"
    return framework


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Evaluation and persistence
# ---------------------------------------------------------------------------

def evaluate_and_save(
    config: dict,
    dataset: DatasetLoader,
    components: dict,
    sources: dict,
    framework: str,
    current_test: str,
    last_llm_log: str,
    success: bool,
    model_tag: str,
    metadata: dict,
    info: list[int],
    mode: str,
    mode_metrics: dict | None = None,
):
    """Evaluate the generated test and save JSON artifacts.

    ``mode_metrics`` contains method-specific values such as iteration counts.
    """
    comparator = components['comparator']
    runner = components['runner']
    file_manager = components['file_manager']
    original_test_path = metadata['test_file_path']
    test_file_before = sources['test_file_before']
    file_model_tag = model_tag
    if mode_metrics:
        file_model_tag = mode_metrics.get('file_model_tag', model_tag)

    
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    ground_truth_test = dataset.get_test_file_after()
    comparison = None
    if ground_truth_test:
        comparison = comparator.compare_tests(current_test, ground_truth_test)
        print(f"\nKey Metrics:")
        print(f"  CodeBLEU:        {comparison['similarity']:.4f}")
        print(f"  Line Count Diff: {comparison['line_count_diff']}")
        print(f"  Char Count Diff: {comparison['char_count_diff']}")

        coverage = comparison['function_coverage']
        print(f"\nFunction Coverage (AST):")
        print(f"  LLM Functions:      {coverage['llm_test_functions']}")
        print(f"  Ground Truth Functions: {coverage['ground_truth_test_functions']}")
        print(f"  Common Functions:    {coverage['common_functions']}")
        if coverage['llm_only']:
            print(f"  LLM Only:           {coverage['llm_only']}")
        if coverage['ground_truth_only']:
            print(f"  Ground Truth Only: {coverage['ground_truth_only']}")
    else:
        print("  Could not get the ground-truth test for comparison.")

    
    print("\n" + "-" * 40)
    print("Test Case Comparison (LLM vs Ground Truth)")
    print("-" * 40)

    llm_run_result = comparator.parse_test_run_result(last_llm_log, framework=framework)
    compilation_result = llm_run_result['compilation']
    print(f"\n  LLM compilation: {compilation_result}")
    print(f"  LLM test result: {llm_run_result['test_result']} "
          f"({llm_run_result['passing_count']} passing, {llm_run_result['failing_count']} failing)")

    
    llm_cases = comparator.parse_passing_tests_from_log(last_llm_log)
    print(f"\n  LLM passing cases ({len(llm_cases)}):")
    for c in llm_cases:
        print(f"    + {c}")

    
    ground_truth_abs_path = file_manager._normalize_path(metadata['test_file_path'])
    ground_truth_rel_path = str(
        ground_truth_abs_path.relative_to(file_manager.project_root)
    )
    cache_key = ground_truth_test_cache_key(config, metadata, framework)
    cached_ground_truth = load_ground_truth_test_cache(config, cache_key)
    ground_truth_from_cache = cached_ground_truth is not None

    if cached_ground_truth:
        print(f"\n  Ground-truth test cache hit: {ground_truth_rel_path}")
        ground_truth_success = bool(cached_ground_truth.get("success"))
        ground_truth_log = str(cached_ground_truth.get("log", ""))
        ground_truth_cases = list(cached_ground_truth.get("passing_cases") or [])
        if not ground_truth_cases and ground_truth_log:
            ground_truth_cases = comparator.parse_passing_tests_from_log(
                ground_truth_log
            )
        ground_truth_run_result = {
            "compilation": cached_ground_truth.get("compilation", "pass"),
            "test_result": cached_ground_truth.get("test_result", "pass"),
            "passing_count": int(
                cached_ground_truth.get("passing_count", len(ground_truth_cases)) or 0
            ),
            "failing_count": int(cached_ground_truth.get("failing_count", 0) or 0),
        }
    else:
        print(f"\n  Running ground-truth test: {ground_truth_rel_path}")
        ground_truth_success, ground_truth_log = runner.run(
            ground_truth_rel_path, framework=framework
        )
        ground_truth_run_result = comparator.parse_test_run_result(
            ground_truth_log, framework=framework
        )
        ground_truth_cases = comparator.parse_passing_tests_from_log(
            ground_truth_log
        )
        if ground_truth_success:
            save_ground_truth_test_cache(
                config,
                cache_key,
                {
                    "record_index": int(config["record_index"]),
                    "repo_name": metadata.get("repo_name", ""),
                    "test_commit_SHA": metadata.get("test_commit_SHA", ""),
                    "test_file_path": metadata.get("test_file_path", ""),
                    "framework": framework,
                    "success": ground_truth_success,
                    "compilation": ground_truth_run_result.get("compilation"),
                    "test_result": ground_truth_run_result.get("test_result"),
                    "passing_count": len(ground_truth_cases),
                    "parsed_passing_count": ground_truth_run_result.get(
                        "passing_count", 0
                    ),
                    "failing_count": ground_truth_run_result.get("failing_count", 0),
                    "passing_cases": ground_truth_cases,
                    "log": ground_truth_log,
                },
            )

    print(
        f"  Ground-truth test "
        f"{'PASSED' if ground_truth_success else 'FAILED'}"
    )
    print(f"  Ground-truth passing cases ({len(ground_truth_cases)}):")
    for c in ground_truth_cases:
        print(f"    + {c}")

    
    case_comparison = comparator.compare_test_case_semantics(
        llm_cases,
        ground_truth_cases,
        current_test,
        ground_truth_test or "",
    )
    print(f"\n  UCR: {case_comparison['ucr']}%")
    if case_comparison.get('match_summary'):
        summary = case_comparison['match_summary']
        print(
            "  Match summary: "
            f"exact={summary.get('exact', 0)}, "
            f"normalized={summary.get('normalized', 0)}, "
            f"semantic={summary.get('semantic', 0)}"
        )
    if case_comparison['llm_only']:
        print(f"  LLM only: {case_comparison['llm_only']}")
    if case_comparison['ground_truth_only']:
        print(f"  Ground truth only: {case_comparison['ground_truth_only']}")

    
    llm_changed = "NO" if current_test.strip() == test_file_before.strip() else "YES"
    print(f"\n  LLM-test changed?: {llm_changed}")

    llm_abs_path = file_manager.get_llm_test_path(original_test_path, model_name=file_model_tag)
    syntax_pass = comparator.check_syntax(str(llm_abs_path))
    print(f"  Syntax check:     {'PASS' if syntax_pass else 'FAIL'}")

    
    project_name = infer_project_name_from_metadata(metadata)
    test_file_stem = Path(metadata['test_file_path']).stem
    # Parallel model workers must reserve the sample output directory and
    # archive their source serially.  Otherwise a worker can observe a sibling
    # model directory before its eval.json exists and incorrectly choose a
    # duplicate suffixed output directory.
    with _OUTPUT_ARCHIVE_LOCK:
        output_test_dir = resolve_output_test_dir(
            mode, project_name, test_file_stem, config["record_index"]
        )
        output_dir = output_test_dir / model_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        # Archive this model's generated source before the outer main script
        # can be interrupted.  The source is deleted only after the copy and
        # metadata have both been committed successfully.
        archive_info = archive_generated_test_source(
            source_path=llm_abs_path,
            destination_dir=output_dir,
            metadata=metadata,
            mode=mode,
            model_name=model_tag,
            original_test_file=original_test_path,
            record_index=int(config["record_index"]),
        )
    if archive_info["archived"]:
        print(f"  Generated test archived: {archive_info['archive_path']}")
    else:
        print(f"  WARNING: generated test was not archived: {archive_info['archive_error']}")

    evaluation = {
        'record_index': config['record_index'],
        'metadata': metadata,
        'mode': mode,
        'llm': model_tag,
        'llm_test_path': str(llm_abs_path.relative_to(file_manager.project_root)),
        'info_enabled': info,
        'final_test_length': len(current_test),
        'test_passed': success,
        'framework': framework,
        'llm_run_result': llm_run_result,
        'syntax_pass': syntax_pass,
        'generated_test_archived': archive_info['archived'],
        'generated_test_archive_path': archive_info['archive_path'],
    }
    if archive_info["archive_error"]:
        evaluation['generated_test_archive_error'] = archive_info['archive_error']

    
    if mode_metrics:
        evaluation.update(mode_metrics)

    if comparison:
        evaluation['comparison'] = comparison
        report = comparator.generate_evaluation_report(comparison, metadata)
        report_path = output_dir / "report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  Report saved to: {report_path}")

    evaluation['case_comparison'] = case_comparison
    evaluation['llm_test_changed'] = llm_changed
    evaluation['ground_truth_test_result'] = {
        'from_cache': ground_truth_from_cache,
        'success': ground_truth_success,
        'compilation': ground_truth_run_result.get('compilation'),
        'test_result': ground_truth_run_result.get('test_result'),
        'passing_count': len(ground_truth_cases),
        'parsed_passing_count': ground_truth_run_result.get('passing_count', 0),
        'failing_count': ground_truth_run_result.get('failing_count', 0),
        'passing_cases': ground_truth_cases,
    }

    eval_path = output_dir / "eval.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    print(f"  Evaluation saved to: {eval_path}")

    
