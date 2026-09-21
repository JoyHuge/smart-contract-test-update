"""
main_SCT.py - SCT-Agent entry point.

This script runs the full SCT-Agent and its AST, Test Runner, and combined
ablations. It uses JSON line patches instead of whole-file generation.
"""

from __future__ import annotations

from agent.sct_agent import SCTAgent, STRICT_INFO_NAMES
from base import (
    create_chat_llm,
    detect_framework,
    evaluate_and_save,
    init_components,
    is_run_completed,
    load_cached_sources,
    print_skip_completed_banner,
    resolve_model_list,
    run_model_jobs,
    save_cached_sources,
    should_skip_completed_run,
    load_llm_environment,
)
from tools.dataset_loader import DatasetLoader
from tools.run_config import apply_runtime_overrides
from tools.sct_ast_parser import SctSolidityAstParser


SUPPORTED_MODES = ("full", "wo_test_runner", "wo_ast_wo_test_runner")
SCT_SOURCE_CACHE_NAME = f"sct_{SctSolidityAstParser.CACHE_TAG}"


def resolve_mode(mode: str) -> dict:
    """Resolve SCT-Agent experiment mode into info/tool settings."""
    normalized = (mode or "").strip().lower().replace("-", "_")
    if normalized == "full":
        return {
            "experiment_name": "SCT-Agent",
            "use_test_runner_feedback": True,
            "info": [3, 4, 6, 9, 12, 13],
        }
    if normalized == "wo_test_runner":
        return {
            "experiment_name": "SCT-Agent-wo-TestRunner",
            "use_test_runner_feedback": False,
            "info": [3, 4, 9, 12, 13],
        }
    if normalized == "wo_ast_wo_test_runner":
        return {
            "experiment_name": "SCT-Agent-wo-AST-wo-TestRunner",
            "use_test_runner_feedback": False,
            "info": [3, 4, 12, 13],
        }
    known = ", ".join(SUPPORTED_MODES)
    raise ValueError(f"Unknown mode. Use one of: {known}.")


def resolve_mode_list(mode_setting) -> list[str]:
    """Resolve config['mode'] into SCT-Agent experiment modes.

    Recommended config forms:
      - "mode": ["full"] for a single mode
      - "mode": ["full", "wo_test_runner"] for selected modes
    """
    if not isinstance(mode_setting, list):
        raise ValueError(
            "config['mode'] must be a list, e.g. ['full'] or "
            "['full', 'wo_test_runner']."
        )

    mode_list = [str(part).strip().lower().replace("-", "_") for part in mode_setting if str(part).strip()]
    if not mode_list:
        raise ValueError("config['mode'] must contain at least one SCT-Agent mode.")

    unknown = [mode for mode in mode_list if mode not in SUPPORTED_MODES]
    if unknown:
        known = ", ".join(SUPPORTED_MODES)
        raise ValueError(
            f"Unknown SCT-Agent mode(s): {unknown}. "
            f"Use a list like ['full'] or ['full', 'wo_test_runner']. "
            f"Known modes: {known}."
        )
    return mode_list


def print_info_status(info: list[int]) -> None:
    print("\nEnabled SCT-Agent info sources:")
    for key, name in STRICT_INFO_NAMES.items():
        status = "ON" if key in info else "OFF"
        print(f"  {key}: {name:28s} {status}")


def collect_sct_sources(config: dict, dataset: DatasetLoader, info: list[int]) -> dict:
    metadata = dataset.get_metadata()
    cached_sources = load_cached_sources(
        config,
        metadata,
        info,
        cache_name=SCT_SOURCE_CACHE_NAME,
    )
    if cached_sources is not None:
        print("\nCollected SCT-Agent sources from cache:")
        print(f"  [3] contract_diff:    {len(cached_sources.get('contract_diff', ''))} chars")
        print(f"  [4] test_file_before: {len(cached_sources.get('test_file_before', ''))} chars")
        print(f"  [9] pnew_ast:         {len(cached_sources.get('pnew_ast', ''))} chars")
        return cached_sources

    contract_diff = dataset.get_contract_diff() if 3 in info else ""
    test_file_before = dataset.get_test_file_before() if 4 in info else ""
    pnew_ast = ""

    if 9 in info:
        parser = SctSolidityAstParser()
        pnew_ast = parser.parse_to_json(dataset.get_contract_file_after(), label="Pnew")

    print("\nCollected SCT-Agent sources:")
    print(f"  [3] contract_diff:    {len(contract_diff)} chars")
    print(f"  [4] test_file_before: {len(test_file_before)} chars")
    print(f"  [9] pnew_ast:         {len(pnew_ast)} chars")

    # evaluate_and_save expects these keys. SCT-Agent fills the selected sources
    # and leaves optional context empty.
    sources = {
        "contract_file_before": "",
        "contract_file_after": "",
        "contract_diff": contract_diff,
        "test_file_before": test_file_before,
        "contract_context": "",
        "all_project_contracts": "",
        "project_config": "",
        "pnew_ast": pnew_ast,
    }
    save_cached_sources(
        config,
        metadata,
        info,
        cache_name=SCT_SOURCE_CACHE_NAME,
        sources=sources,
    )
    return sources


def main():
    config = {
        # Docker supplies all record-specific values through environment variables.
        "LLM": ["gpt"],
        "mode": ["full"],
        "project_dir": "/workspace/project",
        "dataset_path": "/data/dataset.jsonl",
        "test_framework": "",
        "record_index": 0,

        # Agent settings
        "temperature": 0.0,
        "max_iterations": 3,
        "max_output_tokens": 65536,
        "timeout": 1200,
        "llm_invoke_retries": 3,
        "llm_retry_wait": 30,
        "skip_completed": True,
        "parallel_models": False,
        # Information sources are selected by resolve_mode().
    }
    config = apply_runtime_overrides(config)
    print_skip_completed_banner(config)

    try:
        mode_list = resolve_mode_list(config.get("mode", ["full"]))
    except ValueError as exc:
        print(f"\n{exc}")
        return

    dataset = DatasetLoader(config["dataset_path"])
    total_records = dataset.load_dataset()
    if total_records == 0:
        print("No records loaded. Exiting.")
        return

    if not dataset.set_current_record(config["record_index"]):
        print(f"Invalid record_index: {config['record_index']}")
        return

    metadata = dataset.get_metadata()
    print(f"\nProcessing Record {config['record_index']} of {total_records}")
    print(f"  Repo: {metadata['repo_name']}")
    print(f"  Contract: {metadata['production_file_path']}")
    print(f"  Test: {metadata['test_file_path']}")

    validation = dataset.validate_record()
    if validation.get("warnings"):
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")
    if validation.get("errors"):
        print("\nErrors:")
        for error in validation["errors"]:
            print(f"   - {error}")
        return

    components = init_components(config)
    metadata_framework = str(metadata.get("test_framework", "") or "").strip().lower()
    framework = metadata_framework if metadata_framework not in {"", "unknown", "auto", "none", "n/a", "na"} else detect_framework(config, components)
    if not framework:
        framework = "hardhat"
    print(f"\nFramework: {framework}")

    llm_setting = config.get("LLM", [])
    try:
        model_list = resolve_model_list(llm_setting)
    except ValueError as exc:
        print(f"\n{exc}")
        return

    for mode_name in mode_list:
        mode_config = resolve_mode(mode_name)
        info = mode_config["info"]
        experiment_name = mode_config["experiment_name"]
        use_test_runner_feedback = mode_config["use_test_runner_feedback"]
        evaluation_config = dict(config)

        print(f"\n{'=' * 60}")
        print(f"{experiment_name} Mode")
        print(f"{'=' * 60}")
        print(f"Mode: {mode_name}")
        print(f"Use Test Runner feedback: {use_test_runner_feedback}")
        print_info_status(info)

        sources = collect_sct_sources(config, dataset, info)
        if not sources["test_file_before"].strip():
            print("Record has empty test_file_before; cannot start from Told.")
            continue

        def run_model(model_name: str) -> None:
            if should_skip_completed_run(config) and is_run_completed(
                config, metadata, experiment_name, model_name
            ):
                print(f"\n[resume] Skipping {experiment_name} / {model_name} (eval.json exists)")
                return

            model_tag = model_name
            file_model_tag = model_name if len(mode_list) == 1 else f"{model_name}_{mode_name}"
            try:
                llm_settings = load_llm_environment(model_name)
            except ValueError as exc:
                print(f"  {exc}")
                return
            print(f"\n{'=' * 60}")
            print(f"Mode: {experiment_name}")
            print(
                f"LLM: {model_name}  "
                f"(model={llm_settings['model']}, base_url={llm_settings['base_url']})"
            )
            print(f"File tag: {file_model_tag}")
            print(f"{'=' * 60}")

            try:
                llm = create_chat_llm({**config, "LLM": model_name})
            except ValueError as exc:
                print(f"  {exc}")
                return

            file_manager = components["file_manager"]
            original_test_path = metadata["test_file_path"]
            if file_manager.llm_test_exists(original_test_path, model_name=file_model_tag):
                file_manager.delete_llm_test(original_test_path, model_name=file_model_tag)

            agent = SCTAgent(
                llm=llm,
                runner=components["runner"],
                file_manager=file_manager,
                framework=framework,
                original_test_path=original_test_path,
                model_tag=file_model_tag,
                info=info,
                max_iterations=config["max_iterations"],
                use_test_runner_feedback=use_test_runner_feedback,
                llm_invoke_config=config,
            )

            try:
                result = agent.run(
                    initial_test=sources["test_file_before"],
                    sources=sources,
                )

                if not use_test_runner_feedback:
                    print("\n  Final evaluation run (not fed back to LLM)")
                    success, log = components["runner"].run(result.saved_relative_path, framework=framework)
                    result.success = success
                    result.last_log = log

                print("\n" + "-" * 40)
                print("SCT-Agent Result")
                print("-" * 40)
                print(f"  Mode:       {experiment_name}")
                print(f"  Saved test: {result.saved_relative_path}")
                print(f"  Passed:     {result.success}")
                print(f"  Iterations: {result.iterations}")
                print(f"  Wall time:  {result.wall_time:.2f}s")
                timing = result.time_breakdown
                print(
                    "  Timing:     "
                    f"sandbox={timing.get('sandbox_seconds', 0.0):.2f}s, "
                    f"LLM={timing.get('llm_seconds', 0.0):.2f}s, "
                    f"other={timing.get('other_seconds', 0.0):.2f}s"
                )
                print(f"  Tokens:     {result.token_count}")
                print(f"  Patches:    {sum(len(item.get('patches', [])) for item in result.patch_history)}")

                evaluate_and_save(
                    config=evaluation_config,
                    dataset=dataset,
                    components=components,
                    sources=sources,
                    framework=framework,
                    current_test=result.current_test,
                    last_llm_log=result.last_log,
                    success=result.success,
                    model_tag=model_tag,
                    metadata=metadata,
                    info=info,
                    mode=experiment_name,
                    mode_metrics={
                        "iterations": result.iterations,
                        "patch_count": sum(len(item.get("patches", [])) for item in result.patch_history),
                        "patch_history": result.patch_history,
                        "final_judgement": result.final_judgement,
                        "use_test_runner_feedback": use_test_runner_feedback,
                        "ast_enabled": 9 in info,
                        "ast_parser": SctSolidityAstParser.PARSER_NAME if 9 in info else "",
                        "ast_schema_version": SctSolidityAstParser.SCHEMA_VERSION if 9 in info else None,
                        "wall_time": round(result.wall_time, 2),
                        "tokens": result.token_count,
                        "time_breakdown": result.time_breakdown,
                        "file_model_tag": file_model_tag,
                    },
                )
            except Exception as exc:
                print(f"\n  Run failed for {experiment_name} / {model_name}: {type(exc).__name__}: {exc}")
                return

        run_model_jobs(config, model_list, run_model, experiment_name)

    print("\n" + "=" * 60)
    print("SCT-Agent: All mode/model runs done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
