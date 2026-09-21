import os


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> list[str]:
    """Read a comma-separated runtime override without editing main_*.py."""
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def apply_runtime_overrides(config: dict) -> dict:
    """Apply settings passed by the run script through environment variables."""
    updated = dict(config)

    if os.getenv("SCT_RECORD_INDEX"):
        updated["record_index"] = int(os.environ["SCT_RECORD_INDEX"])

    models = _env_csv("SCT_MODELS")
    if models:
        updated["LLM"] = models

    sct_modes = _env_csv("SCT_SCT_MODES")
    if sct_modes:
        updated["mode"] = sct_modes

    if os.getenv("SCT_PROJECT_DIR"):
        updated["project_dir"] = os.environ["SCT_PROJECT_DIR"]

    if os.getenv("SCT_DATASET_PROJECT_PREFIX"):
        updated["dataset_project_prefix"] = os.environ["SCT_DATASET_PROJECT_PREFIX"]

    if os.getenv("SCT_DATASET_PATH"):
        updated["dataset_path"] = os.environ["SCT_DATASET_PATH"]

    if os.getenv("SCT_TEST_FRAMEWORK"):
        updated["test_framework"] = os.environ["SCT_TEST_FRAMEWORK"]

    if os.getenv("SCT_SOURCES_CACHE_DIR"):
        updated["sources_cache_dir"] = os.environ["SCT_SOURCES_CACHE_DIR"]

    if os.getenv("SCT_LLM_TIMEOUT"):
        updated["timeout"] = float(os.environ["SCT_LLM_TIMEOUT"])

    if os.getenv("SCT_LLM_CONNECT_TIMEOUT"):
        updated["connect_timeout"] = float(os.environ["SCT_LLM_CONNECT_TIMEOUT"])

    if os.getenv("SCT_LLM_WRITE_TIMEOUT"):
        updated["write_timeout"] = float(os.environ["SCT_LLM_WRITE_TIMEOUT"])

    if os.getenv("SCT_LLM_POOL_TIMEOUT"):
        updated["pool_timeout"] = float(os.environ["SCT_LLM_POOL_TIMEOUT"])

    if os.getenv("SCT_LLM_INVOKE_RETRIES"):
        updated["llm_invoke_retries"] = int(os.environ["SCT_LLM_INVOKE_RETRIES"])

    if os.getenv("SCT_LLM_RETRY_WAIT"):
        updated["llm_retry_wait"] = int(os.environ["SCT_LLM_RETRY_WAIT"])

    if os.getenv("SCT_LLM_STREAMING"):
        updated["streaming"] = _env_flag("SCT_LLM_STREAMING")

    if os.getenv("SCT_LLM_STREAM_USAGE"):
        updated["stream_usage"] = _env_flag("SCT_LLM_STREAM_USAGE")

    if os.getenv("SCT_LLM_MAX_OUTPUT_TOKENS"):
        max_output_tokens = int(os.environ["SCT_LLM_MAX_OUTPUT_TOKENS"])
        if max_output_tokens <= 0:
            raise ValueError("SCT_LLM_MAX_OUTPUT_TOKENS must be greater than 0")
        updated["max_output_tokens"] = max_output_tokens

    if os.getenv("SCT_PARALLEL_MODELS"):
        updated["parallel_models"] = _env_flag("SCT_PARALLEL_MODELS")

    # Resume / force-rerun priority: force, explicit skip flag, then config default.
    if os.getenv("SCT_FORCE_RERUN"):
        if _env_flag("SCT_FORCE_RERUN"):
            updated["force_rerun"] = True
            updated["skip_completed"] = False
        else:
            updated["force_rerun"] = False

    if os.getenv("SCT_SKIP_COMPLETED"):
        updated["skip_completed"] = _env_flag("SCT_SKIP_COMPLETED")
    elif os.getenv("SCT_RESUME"):
        updated["skip_completed"] = _env_flag("SCT_RESUME")

    return updated
