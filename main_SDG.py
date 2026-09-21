"""Static Direct Generation: generate once, execute, and evaluate."""

from base import (
    create_chat_llm, print_info_status,
    collect_info_sources, init_components, generate_test,
    detect_framework, evaluate_and_save, resolve_model_list,
    is_run_completed, print_skip_completed_banner, should_skip_completed_run,
    run_model_jobs, load_llm_environment,
)
from tools.dataset_loader import DatasetLoader
from tools.run_config import apply_runtime_overrides


def main():
    config = {
        # Docker supplies all record-specific values through environment variables.
        "LLM": ["gpt"],
        "project_dir": "/workspace/project",
        "dataset_path": "/data/dataset.jsonl",
        "test_framework": "",
        "record_index": 0,
        "temperature": 0.0,
        "max_output_tokens": 65536,
        "timeout": 1200,
        "llm_invoke_retries": 3,
        "llm_retry_wait": 30,
        "skip_completed": True,
        "parallel_models": False,
        # Static Direct Generation uses the paper baseline's static inputs.
        "info": [3, 4],
    }
    config = apply_runtime_overrides(config)
    print_skip_completed_banner(config)

    info = config['info']

    print("=" * 60)
    print("SDG Mode (Static Direct Generation)")
    print("=" * 60)
    print_info_status(info)

    
    dataset = DatasetLoader(config['dataset_path'])
    total_records = dataset.load_dataset()
    if total_records == 0:
        print("No records loaded. Exiting.")
        return

    if not dataset.set_current_record(config['record_index']):
        print(f"Invalid record_index: {config['record_index']}")
        return
    metadata = dataset.get_metadata()

    print(f"\nProcessing Record {config['record_index']} of {total_records}")
    print(f"  Repo: {metadata['repo_name']}")
    print(f"  Contract: {metadata['production_file_path']}")
    print(f"  Test: {metadata['test_file_path']}")

    validation = dataset.validate_record()
    if validation.get('warnings'):
        print("\nWarnings:")
        for w in validation['warnings']:
            print(f"   - {w}")
    if validation.get('errors'):
        print("\nErrors:")
        for error in validation['errors']:
            print(f"   - {error}")
        return

    components = init_components(config)
    sources = collect_info_sources(config, dataset, info)
    framework = detect_framework(config, components)

    
    llm_setting = config.get("LLM", [])
    try:
        model_list = resolve_model_list(llm_setting)
    except ValueError as e:
        print(f"\n{e}")
        return

    
    def run_model(model_name: str) -> None:
        if should_skip_completed_run(config) and is_run_completed(config, metadata, "SDG", model_name):
            print(f"\n[resume] Skipping SDG / {model_name} (eval.json exists)")
            return

        model_tag = model_name
        try:
            llm_settings = load_llm_environment(model_name)
        except ValueError as e:
            print(f"  {e}")
            return
        print(f"\n{'=' * 60}")
        print(
            f"LLM: {model_name}  "
            f"(model={llm_settings['model']}, base_url={llm_settings['base_url']})"
        )
        print(f"{'=' * 60}")

        try:
            llm = create_chat_llm({**config, "LLM": model_name})
        except ValueError as e:
            print(f"  {e}")
            return

        
        print("\n" + "-" * 40)
        print("Generate Test (single attempt, no refinement)")
        print("-" * 40)
        test_code = generate_test(llm, config, info, sources)
        if not test_code.strip():
            print(f"  Skipping {model_name} due to empty generation.")
            return
        print(f"  Generated: {len(test_code)} chars")

        
        file_manager = components['file_manager']
        original_test_path = metadata['test_file_path']

        if file_manager.llm_test_exists(original_test_path, model_name=model_tag):
            file_manager.delete_llm_test(original_test_path, model_name=model_tag)
        try:
            saved_path = file_manager.save_llm_test(original_test_path, test_code, model_name=model_tag)
            print(f"  Saved: {saved_path}")
        except Exception as e:
            print(f"  Failed to save: {e}")
            return

        
        print("\n" + "-" * 40)
        print("Run Test (single run, no fix)")
        print("-" * 40)

        runner = components['runner']
        llm_test_path = file_manager.get_llm_test_path(original_test_path, model_name=model_tag)
        relative_path = str(llm_test_path.relative_to(file_manager.project_root))

        print(f"  Running: {relative_path}")
        success, log = runner.run(relative_path, framework=framework)
        print(f"  Result: {'PASSED' if success else 'FAILED'}")
        if not success:
            log_lines = log.strip().split('\n')
            for line in log_lines[-10:]:
                print(f"    {line}")

        
        evaluate_and_save(
            config=config,
            dataset=dataset,
            components=components,
            sources=sources,
            framework=framework,
            current_test=test_code,
            last_llm_log=log,
            success=success,
            model_tag=model_tag,
            metadata=metadata,
            info=info,
            mode="SDG",
            mode_metrics={
                'iterations': 1,
            },
        )

    run_model_jobs(config, model_list, run_model, "SDG")

    print("\n" + "=" * 60)
    print("SDG: All models done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
