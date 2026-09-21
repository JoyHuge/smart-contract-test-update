"""Static Self-Refine: generate, review iteratively, execute, and evaluate."""

import json
import re

from base import (
    create_chat_llm, print_info_status,
    collect_info_sources, init_components, generate_test,
    detect_framework, extract_code_from_llm, build_section,
    evaluate_and_save, resolve_model_list, invoke_llm_with_retry,
    is_run_completed, print_skip_completed_banner, should_skip_completed_run,
    run_model_jobs, load_llm_environment,
)
from tools.dataset_loader import DatasetLoader
from tools.run_config import apply_runtime_overrides


# ---------------------------------------------------------------------------
# SSR self-review prompt
# ---------------------------------------------------------------------------

def build_self_review_prompt(info: list[int], test_content: str, **kwargs) -> str:
    """Build the SSR prompt for a deep logical review rather than a syntax check."""
    prompt = """You are a senior smart-contract test reviewer performing a deep logical review.

You previously generated an updated test file for a smart contract that has been modified. Now review your own work critically.

=== TEST FILE TO REVIEW ===
"""
    prompt += test_content + "\n"

    
    sections = [
        (1, "CONTRACT FILE BEFORE CHANGE (OLD version)", kwargs.get('contract_file_before', '')),
        (2, "CONTRACT FILE AFTER CHANGE (NEW target version)", kwargs.get('contract_file_after', '')),
        (3, "CONTRACT DIFF (what changed)", kwargs.get('contract_diff', '')),
        (5, "CURRENT CONTRACT CODE (referenced by test)", kwargs.get('contract_context', '')),
        (7, "ALL PROJECT CONTRACT CODE (authoritative API context)", kwargs.get('all_project_contracts', '')),
        (8, "PROJECT CONFIG AND UTILITIES", kwargs.get('project_config', '')),
    ]
    for info_id, header, content in sections:
        prompt += build_section(info_id, header, content, info)

    prompt += """
REVIEW DIMENSIONS — check each one carefully:

1. SYNTAX & STRUCTURE
   - Are all `artifacts.require()` calls correct and referencing existing contracts?
   - Are all `contract()` / `describe()` / `it()` blocks properly closed?
   - Are braces and parentheses balanced?

2. CONTRACT CHANGE ADAPTATION
   - Does the test reflect ALL changes shown in CONTRACT DIFF?
   - For modified functions: are the test inputs and expected outputs updated to match the NEW contract behavior?
   - For added functions: are there tests covering the new functionality?
   - For deleted functions: are the corresponding test cases removed?

3. ASSERTION CORRECTNESS (most critical)
   - Compare each assertion's expected value against the actual behavior defined in CONTRACT FILE AFTER CHANGE.
   - Are return value types correct (e.g., BigNumber vs number, bytes vs string)?
   - Are event assertions checking the correct parameter names and types?
   - Are error/revert assertions testing the correct revert conditions?

4. TEST COVERAGE
   - Do the tests cover both normal paths and edge cases for changed functions?
   - Are `beforeEach` / setup blocks correctly initializing state for the new contract version?
   - Are there missing tests for new public/external functions?

5. FRAMEWORK COMPATIBILITY
   - Are test setup patterns (deploy, get accounts, etc.) consistent with the project's framework?
   - Are helper functions and imports correct?

OUTPUT FORMAT — respond in TWO parts:

Part 1: A short JSON wrapped in ```json ... ```:
```json
{
  "pass": true or false,
  "issues": ["list of specific issues found, empty if pass"]
}
```

Part 2 (only if pass=false): The COMPLETE revised test file in a separate ```javascript ... ``` block.

Example for pass=false:
```json
{
  "pass": false,
  "issues": ["Assertion expected value mismatch on line 45"]
}
```

```javascript
// COMPLETE revised test file here
```

Example for pass=true:
```json
{
  "pass": true,
  "issues": []
}
```

RULES:
- If the test file looks correct after your review, set pass=true. Do NOT invent issues or make speculative changes.
- If pass=true, output ONLY the JSON. Do NOT output any code block.
- If pass=false, output the JSON followed by the COMPLETE revised test file in ```javascript ... ```.
- Only set pass=false when you find CONCRETE, VERIFIABLE errors (wrong assertion values, missing await, incorrect function signatures, etc.).
- If you are unsure whether something is actually wrong, leave it as-is and set pass=true. Unnecessary modifications are more harmful than minor imperfections.
- When modifying, preserve the original file structure as much as possible. Only change the specific parts that have confirmed errors.
"""
    return prompt


def parse_review_result(llm_output: str) -> dict:
    """
    Parse an LLM self-review whose JSON decision and revised code are separate.

    Return ``{"pass": bool, "issues": list, "revised_test": str}``.
    """
    default = {"pass": True, "issues": [], "revised_test": ""}

    
    json_match = re.search(r'```json\s*\n(.*?)```', llm_output, re.DOTALL)
    if not json_match:
        
        json_match = re.search(r'\{\s*"pass"\s*:\s*(true|false)\s*,\s*"issues"\s*:.*?\}', llm_output, re.DOTALL)

    is_pass = None
    issues = []

    if json_match:
        try:
            json_str = json_match.group(1) if json_match.lastindex == 1 else json_match.group(0)
            result = json.loads(json_str)
            is_pass = result.get("pass", None)
            issues = result.get("issues", [])
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"    Failed to parse review JSON: {e}")

    
    if is_pass is None:
        output_lower = llm_output.lower()
        if '"pass": true' in output_lower or '"pass":true' in output_lower:
            is_pass = True
        elif '"pass": false' in output_lower or '"pass":false' in output_lower:
            is_pass = False
        else:
            print("    Could not parse review result — treating as PASS")
            return default

    if is_pass:
        return {"pass": True, "issues": issues, "revised_test": ""}

    
    revised_test = extract_code_from_llm(llm_output)

    if revised_test.strip():
        return {"pass": False, "issues": issues, "revised_test": revised_test}
    else:
        print("    Review said FAIL but no revised code found — treating as PASS")
        return default


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    config = {
        # Docker supplies all record-specific values through environment variables.
        "LLM": ["gpt"],
        "project_dir": "/workspace/project",
        "dataset_path": "/data/dataset.jsonl",
        "test_framework": "",
        "record_index": 0,
        "temperature": 0.0,
        "max_iterations": 3,
        "max_output_tokens": 65536,
        "timeout": 1200,
        "llm_invoke_retries": 3,
        "llm_retry_wait": 30,
        "skip_completed": True,
        "parallel_models": False,
        # Static Self-Refine uses the paper baseline's static inputs.
        "info": [3, 4],
    }
    config = apply_runtime_overrides(config)
    print_skip_completed_banner(config)

    info = config['info']

    print("=" * 60)
    print("SSR Mode (Static Self-Refine)")
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
        if should_skip_completed_run(config) and is_run_completed(config, metadata, "SSR", model_name):
            print(f"\n[resume] Skipping SSR / {model_name} (eval.json exists)")
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
        print("Stage 1: Generate")
        print("-" * 40)
        test_code = generate_test(llm, config, info, sources)
        if not test_code.strip():
            print(f"  Skipping {model_name} due to empty generation.")
            return
        print(f"  Generated: {len(test_code)} chars")

        
        print("\n" + "-" * 40)
        print("Stage 2: Self-Refine Loop")
        print("-" * 40)

        current_test = test_code
        max_iterations = config.get("max_iterations", 3)
        iterations = 0
        refine_passed = False

        for iteration in range(1, max_iterations + 1):
            print(f"\n  Self-Review round {iteration}/{max_iterations}")
            iterations = iteration

            
            review_prompt = build_self_review_prompt(
                info,
                test_content=current_test,
                contract_file_before=sources['contract_file_before'],
                contract_file_after=sources['contract_file_after'],
                contract_diff=sources['contract_diff'],
                contract_context=sources['contract_context'],
                all_project_contracts=sources['all_project_contracts'],
                project_config=sources['project_config'],
            )

            try:
                review_response = invoke_llm_with_retry(
                    llm, [("user", review_prompt)], config
                )
                review_result = parse_review_result(review_response.content)
            except Exception as e:
                print(f"    Review failed: {e}")
                break

            if review_result['issues']:
                print(f"    Issues found: {review_result['issues'][:3]}{'...' if len(review_result['issues']) > 3 else ''}")

            if review_result['pass']:
                print(f"    PASS — test is logically sound")
                refine_passed = True
                break
            else:
                
                revised = review_result['revised_test']
                if revised.strip():
                    current_test = revised
                    print(f"    FAIL — revised test: {len(current_test)} chars")
                    iterations = iteration
                else:
                    print(f"    FAIL — no revised code, keeping current version")
                    refine_passed = True
                    break
        else:
            
            print(f"\n  All {max_iterations} self-review rounds completed, using final version")
            iterations = max_iterations

        
        file_manager = components['file_manager']
        original_test_path = metadata['test_file_path']

        if file_manager.llm_test_exists(original_test_path, model_name=model_tag):
            file_manager.delete_llm_test(original_test_path, model_name=model_tag)
        try:
            saved_path = file_manager.save_llm_test(original_test_path, current_test, model_name=model_tag)
            print(f"\n  Saved final test: {saved_path}")
        except Exception as e:
            print(f"  Failed to save: {e}")
            return

        
        print("\n" + "-" * 40)
        print("Stage 3: Run Test")
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
            current_test=current_test,
            last_llm_log=log,
            success=success,
            model_tag=model_tag,
            metadata=metadata,
            info=info,
            mode="SSR",
            mode_metrics={
                'iterations': iterations,
                'refine_passed': refine_passed,
            },
        )

    run_model_jobs(config, model_list, run_model, "SSR")

    print("\n" + "=" * 60)
    print("SSR: All models done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
