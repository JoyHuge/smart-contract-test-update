"""
sct_agent.py - SCT-Agent loop and ablation variants.

This implementation follows the SCT-Agent workflow:
Told -> TestRunner -> ExecutionJudge -> LLM JSON line patch -> ApplyPatch -> repeat.
It deliberately avoids whole-file generation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from base import invoke_llm_with_retry
from tools.execution_judge import ExecutionJudge
from tools.line_patch import LinePatchApplicator


STRICT_INFO_NAMES = {
    3: "contract_diff",
    4: "test_file_before",
    6: "error_log",
    9: "pnew_ast",
    12: "numbered_current_test",
    13: "patch_history",
}


@dataclass
class AgentRunResult:
    success: bool
    current_test: str
    last_log: str
    iterations: int
    patch_history: list[dict[str, Any]] = field(default_factory=list)
    final_judgement: dict[str, Any] = field(default_factory=dict)
    saved_relative_path: str = ""
    wall_time: float = 0.0
    token_count: int = 0
    time_breakdown: dict[str, Any] = field(default_factory=dict)


class SCTAgent:
    """Execution-feedback-driven SCT-Agent with line-level patch actions."""

    def __init__(
        self,
        llm,
        runner,
        file_manager,
        framework: str,
        original_test_path: str,
        model_tag: str,
        info: list[int],
        max_iterations: int = 3,
        use_test_runner_feedback: bool = True,
        llm_invoke_config: dict | None = None,
    ):
        self.llm = llm
        self.runner = runner
        self.file_manager = file_manager
        self.framework = framework
        self.original_test_path = original_test_path
        self.model_tag = model_tag
        self.info = info
        self.max_iterations = max(1, int(max_iterations))
        self.use_test_runner_feedback = use_test_runner_feedback
        self.llm_invoke_config = llm_invoke_config or {}
        self.judge = ExecutionJudge()
        self.patcher = LinePatchApplicator()
        self.token_count = 0
        self._token_warning_printed = False
        self._reset_timing()

    @property
    def ast_enabled(self) -> bool:
        return 9 in self.info

    def _validate_ast_boundary(self, sources: dict[str, str]) -> None:
        """Fail closed if an ablation receives AST data or an AST mode lacks it."""
        pnew_ast = sources.get("pnew_ast", "")
        if not isinstance(pnew_ast, str):
            raise TypeError("sources['pnew_ast'] must be a string.")
        has_ast_payload = bool(pnew_ast.strip())
        if self.ast_enabled and not has_ast_payload:
            raise ValueError("AST-enabled SCT-Agent requires a non-empty Pnew AST payload.")
        if not self.ast_enabled and has_ast_payload:
            raise ValueError("w/o AST mode received a non-empty Pnew AST payload.")

    def _contract_update_evidence(self) -> str:
        if self.ast_enabled:
            return "CONTRACT DIFF and PNEW TREE-SITTER AST"
        return "CONTRACT DIFF"

    def _baseline_pass_judgement(self) -> dict[str, Any]:
        update_evidence = self._contract_update_evidence()
        return {
            "status": "baseline_pass",
            "phase": "initial_update_required",
            "message": (
                "The baseline test passes and there is no runtime error log. "
                f"This is the forced initial update round: inspect {update_evidence} "
                "and the current test, then emit JSON line patches if "
                "the test should be updated for the changed behavior."
            ),
            "stack_excerpt": "",
        }

    def _reset_timing(self) -> None:
        """Reset the per-trajectory timing ledger.

        ``wall_time`` has historically meant active SCT-Agent time: the time
        inside ``run()`` after excluding any TestRunner concurrency-lock wait.
        The three fields below use exactly that same boundary, so their ratios
        are meaningful even when the caller enables parallel execution.
        """
        self._sandbox_seconds = 0.0
        self._llm_seconds = 0.0
        self._sandbox_calls: list[dict[str, Any]] = []
        self._llm_calls: list[dict[str, Any]] = []

    def _lock_wait_seconds(self) -> float:
        if not hasattr(self.runner, "get_lock_wait_time"):
            return 0.0
        try:
            return max(0.0, float(self.runner.get_lock_wait_time()))
        except (TypeError, ValueError):
            return 0.0

    def _run_sandbox(
        self,
        test_file: str,
        *,
        stage: str,
    ) -> tuple[bool, str]:
        """Run one TestRunner call and add only its active time to the ledger."""
        lock_before = self._lock_wait_seconds()
        started = time.perf_counter()
        success: bool | None = None
        raw_log = ""
        error: Exception | None = None
        try:
            success, raw_log = self.runner.run(test_file, framework=self.framework)
            return success, raw_log
        except Exception as exc:
            error = exc
            raise
        finally:
            elapsed = time.perf_counter() - started
            lock_wait = max(0.0, self._lock_wait_seconds() - lock_before)
            active_seconds = max(0.0, elapsed - lock_wait)
            self._sandbox_seconds += active_seconds
            event: dict[str, Any] = {
                "stage": stage,
                "test_file": test_file,
                "elapsed_seconds": round(elapsed, 6),
                "lock_wait_seconds": round(lock_wait, 6),
                "active_seconds": round(active_seconds, 6),
            }
            if success is not None:
                event["success"] = bool(success)
            if error is not None:
                event["error"] = f"{type(error).__name__}: {error}"
            self._sandbox_calls.append(event)

    def _time_breakdown(self, total_active_seconds: float, lock_wait_seconds: float) -> dict[str, Any]:
        """Return a reconciled timing record suitable for ``eval.json``."""
        other_raw = total_active_seconds - self._sandbox_seconds - self._llm_seconds
        other_seconds = max(0.0, other_raw)
        return {
            "schema_version": 1,
            "measurement_scope": (
                "SCTAgent.run() active wall-clock time; excludes TestRunner "
                "concurrency-lock waiting and excludes preflight/container setup."
            ),
            "total_active_seconds": round(total_active_seconds, 6),
            "sandbox_seconds": round(self._sandbox_seconds, 6),
            "llm_seconds": round(self._llm_seconds, 6),
            "other_seconds": round(other_seconds, 6),
            "lock_wait_seconds": round(lock_wait_seconds, 6),
            "reconciliation_error_seconds": round(other_raw - other_seconds, 9),
            "sandbox_call_count": len(self._sandbox_calls),
            "llm_call_count": len(self._llm_calls),
            "sandbox_calls": self._sandbox_calls,
            "llm_calls": self._llm_calls,
        }

    def run(self, initial_test: str, sources: dict[str, str]) -> AgentRunResult:
        self._validate_ast_boundary(sources)
        self.token_count = 0
        self._token_warning_printed = False
        self._reset_timing()
        if hasattr(self.runner, "reset_lock_wait_time"):
            self.runner.reset_lock_wait_time()
        start_time = time.perf_counter()
        if self.use_test_runner_feedback:
            result = self._run_execution_feedback_loop(initial_test, sources)
        else:
            result = self._run_static_review_patch_loop(initial_test, sources)
        elapsed = time.perf_counter() - start_time
        lock_wait = self._lock_wait_seconds()
        wall_time = max(0.0, elapsed - lock_wait)
        result.wall_time = wall_time
        result.token_count = self.token_count
        result.time_breakdown = self._time_breakdown(wall_time, lock_wait)
        return result

    def _run_execution_feedback_loop(self, initial_test: str, sources: dict[str, str]) -> AgentRunResult:
        current_test = initial_test
        saved_relative_path = self._save_current_test(current_test, overwrite=True)
        patch_history: list[dict[str, Any]] = []
        last_log = ""
        final_judgement: dict[str, Any] = {}
        attempts = 0

        for iteration in range(0, self.max_iterations):
            print(f"\n  Iteration {iteration}/{self.max_iterations}: run current test")
            print(f"  Run current test: {saved_relative_path}")
            success, raw_log = self._run_sandbox(
                saved_relative_path,
                stage=f"feedback_iteration_{iteration}",
            )
            last_log = raw_log
            final_judgement = self.judge.judge(success, raw_log, framework=self.framework)

            if success and iteration != 0:
                print(f"  PASS at iteration {iteration}")
                return AgentRunResult(
                    success=True,
                    current_test=current_test,
                    last_log=last_log,
                    iterations=attempts,
                    patch_history=patch_history,
                    final_judgement=final_judgement,
                    saved_relative_path=saved_relative_path,
                )

            if iteration == 0:
                if success:
                    print("  Baseline test PASS; forced initial update still required")
                    final_judgement = self._baseline_pass_judgement()
                else:
                    print(f"  Baseline failed ({final_judgement.get('phase', 'unknown')}); asking LLM for initial patch.")
            else:
                print(f"  FAIL ({final_judgement.get('phase', 'unknown')})")

            message = final_judgement.get("message") or final_judgement.get("stack_excerpt", "")
            if message:
                for line in str(message).splitlines()[:8]:
                    print(f"    {line[:180]}")

            prompt = self._build_patch_prompt(
                sources=sources,
                current_test=current_test,
                judgement=final_judgement,
                patch_history=patch_history,
            )
            response = self._invoke_llm([("user", prompt)])
            attempts += 1
            raw_response = response.content or ""
            try:
                patches = self.patcher.parse_patch_response(raw_response)
                patch_result = self.patcher.apply(current_test, patches)
            except Exception as exc:
                print(f"  Patch response invalid: {type(exc).__name__}: {exc}")
                if iteration == 0 and success:
                    print("  Forced initial update failed, but baseline passed; returning unchanged test.")
                    return AgentRunResult(
                        success=True,
                        current_test=current_test,
                        last_log=last_log,
                        iterations=attempts,
                        patch_history=patch_history,
                        final_judgement={
                            **final_judgement,
                            "status": "initial_update_failed",
                            "message": str(exc),
                        },
                        saved_relative_path=saved_relative_path,
                    )
                print("  Could not obtain an applicable patch; stopping.")
                break

            if not patch_result.ok:
                print(f"  Patch could not be applied: {patch_result.message}")
                if iteration == 0 and success:
                    print("  Forced initial update failed, but baseline passed; returning unchanged test.")
                    return AgentRunResult(
                        success=True,
                        current_test=current_test,
                        last_log=last_log,
                        iterations=attempts,
                        patch_history=patch_history,
                        final_judgement={
                            **final_judgement,
                            "status": "initial_update_failed",
                            "message": patch_result.message,
                        },
                        saved_relative_path=saved_relative_path,
                    )
                print("  Could not obtain an applicable patch; stopping.")
                break

            current_test = patch_result.text
            saved_relative_path = self._save_current_test(current_test, overwrite=True)
            patch_history.append(
                {
                    "iteration": iteration,
                    "attempt": attempts,
                    "forced_initial_update": iteration == 0,
                    "patches": patches,
                    "judgement": final_judgement,
                    "syntax_ok": patch_result.syntax_ok,
                    "syntax_log": patch_result.syntax_log,
                }
            )
            print(f"  Applied {len(patches)} patch(es)")
            if not patch_result.syntax_ok:
                print("  Syntax check failed after patch; next iteration will use runner feedback.")

        print("\n  Final run after stopping the feedback loop")
        print(f"  Run final test: {saved_relative_path}")
        success, raw_log = self._run_sandbox(
            saved_relative_path,
            stage="final_validation",
        )
        last_log = raw_log
        final_judgement = self.judge.judge(success, raw_log, framework=self.framework)

        return AgentRunResult(
            success=success,
            current_test=current_test,
            last_log=last_log,
            iterations=attempts,
            patch_history=patch_history,
            final_judgement=final_judgement,
            saved_relative_path=saved_relative_path,
        )

    def _run_static_review_patch_loop(self, initial_test: str, sources: dict[str, str]) -> AgentRunResult:
        """Run SCT-Agent w/o Test Runner: static self-review + JSON line patches."""
        current_test = initial_test
        saved_relative_path = self._save_current_test(current_test, overwrite=True)
        patch_history: list[dict[str, Any]] = []
        final_judgement: dict[str, Any] = {
            "status": "not_run",
            "phase": "static_review",
            "message": "Test Runner feedback disabled during agent loop.",
        }
        patches_applied = 0

        while patches_applied < self.max_iterations:
            next_version = patches_applied + 1
            print(f"\n  Static review before generating version {next_version}/{self.max_iterations}")
            prompt = self._build_static_review_patch_prompt(
                sources=sources,
                current_test=current_test,
                patch_history=patch_history,
            )
            response = self._invoke_llm([("user", prompt)])
            raw_response = response.content or ""

            try:
                review = self._parse_static_review_response(raw_response)
            except Exception as exc:
                print(f"  Static review response invalid: {type(exc).__name__}: {exc}")
                break

            issues = review.get("issues", [])
            if issues:
                print(f"  Issues: {issues[:3]}{'...' if len(issues) > 3 else ''}")

            patches = review.get("patches", [])
            if review.get("pass") and not patches:
                print("  Static review PASS; no further patch needed.")
                final_judgement = {
                    "status": "static_pass",
                    "phase": "static_review",
                    "issues": issues,
                }
                break

            if not patches:
                print("  Static review found no applicable patch; stopping.")
                final_judgement = {
                    "status": "static_stop",
                    "phase": "static_review",
                    "issues": issues,
                }
                break

            patch_result = self.patcher.apply(current_test, patches)
            if not patch_result.ok:
                print(f"  Patch could not be applied: {patch_result.message}")
                final_judgement = {
                    "status": "patch_failed",
                    "phase": "static_review",
                    "issues": issues,
                    "message": patch_result.message,
                }
                break

            current_test = patch_result.text
            saved_relative_path = self._save_current_test(current_test, overwrite=True)
            patches_applied = next_version
            patch_history.append(
                {
                    "iteration": patches_applied,
                    "issues": issues,
                    "patches": patches,
                    "syntax_ok": patch_result.syntax_ok,
                    "syntax_log": patch_result.syntax_log,
                }
            )
            print(f"  Generated version {patches_applied}/{self.max_iterations} with {len(patches)} patch(es)")
            if not patch_result.syntax_ok:
                print("  Syntax check failed after patch; continuing static review without runner feedback.")

        if patches_applied >= self.max_iterations:
            final_judgement = {
                "status": "max_versions_reached",
                "phase": "static_review",
                "message": "Generated the maximum number of static patched test versions.",
            }

        return AgentRunResult(
            success=False,
            current_test=current_test,
            last_log="",
            iterations=patches_applied,
            patch_history=patch_history,
            final_judgement=final_judgement,
            saved_relative_path=saved_relative_path,
        )

    def _save_current_test(self, current_test: str, overwrite: bool) -> str:
        if self.file_manager.llm_test_exists(self.original_test_path, model_name=self.model_tag):
            overwrite = True
        return self.file_manager.save_llm_test(
            self.original_test_path,
            current_test,
            overwrite=overwrite,
            model_name=self.model_tag,
        )

    def _invoke_llm(self, messages):
        started = time.perf_counter()
        response = None
        error: Exception | None = None
        try:
            response = invoke_llm_with_retry(self.llm, messages, self.llm_invoke_config)
        except Exception as exc:
            error = exc
            raise
        finally:
            elapsed = time.perf_counter() - started
            self._llm_seconds += elapsed
            event: dict[str, Any] = {
                "elapsed_seconds": round(elapsed, 6),
                "message_count": len(messages),
            }
            if error is not None:
                event["error"] = f"{type(error).__name__}: {error}"
            self._llm_calls.append(event)

        if response is None:
            raise RuntimeError("LLM invocation returned no response.")
        total_tokens = self._extract_total_tokens(response)
        if total_tokens is None:
            if not self._token_warning_printed:
                print("  Token usage unavailable in LLM response; Tokens will be recorded as 0.")
                self._token_warning_printed = True
        else:
            self.token_count += total_tokens
            print(f"  LLM tokens: +{total_tokens} (total {self.token_count})")
        return response

    @staticmethod
    def _extract_total_tokens(response) -> int | None:
        usage = getattr(response, "usage_metadata", None) or {}
        total = SCTAgent._usage_value(usage, "total_tokens")
        if total is not None:
            return total

        metadata = getattr(response, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
        total = SCTAgent._usage_value(token_usage, "total_tokens")
        if total is not None:
            return total

        input_tokens = SCTAgent._usage_value(usage, "input_tokens")
        output_tokens = SCTAgent._usage_value(usage, "output_tokens")
        if input_tokens is not None and output_tokens is not None:
            return input_tokens + output_tokens

        prompt_tokens = SCTAgent._usage_value(token_usage, "prompt_tokens")
        completion_tokens = SCTAgent._usage_value(token_usage, "completion_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            return prompt_tokens + completion_tokens

        return None

    @staticmethod
    def _usage_value(usage, key: str) -> int | None:
        if not isinstance(usage, dict) or key not in usage or usage[key] is None:
            return None
        try:
            return int(usage[key])
        except (TypeError, ValueError):
            return None

    def _build_patch_prompt(
        self,
        sources: dict[str, str],
        current_test: str,
        judgement: dict[str, Any],
        patch_history: list[dict[str, Any]],
    ) -> str:
        self._validate_ast_boundary(sources)
        update_evidence = self._contract_update_evidence()
        sections = []
        sections.append(
            "You are SCT-Agent, a smart-contract test co-evolution agent.\n"
            "Use execution feedback and Solidity interface perception "
            "to emit a precise line-level patch for the obsolete JS/TS test. "
            "If the feedback says this is the forced initial update round, "
            f"there may be no runtime error log; use {update_evidence} "
            "to decide the initial test update. Do not rewrite the whole test file."
        )

        if 3 in self.info:
            sections.append(self._section("CONTRACT DIFF Pold -> Pnew", sources.get("contract_diff", "")))
        if 9 in self.info:
            sections.append(
                self._section(
                    "PNEW TREE-SITTER AST / NEW SIGNATURES, TYPES, AND MODIFIERS",
                    sources.get("pnew_ast", ""),
                )
            )
        if 6 in self.info:
            sections.append(self._section("EXECUTION ERROR TRACE", self.judge.to_prompt_text(judgement)))
        if 13 in self.info:
            history_for_prompt = self._compact_patch_history(patch_history)
            sections.append(self._section("PATCH HISTORY", history_for_prompt))
        if 12 in self.info:
            numbered = self.patcher.add_line_numbers(current_test)
            sections.append(self._section("NUMBERED CURRENT TEST FILE", numbered))

        sections.append(
            "TASK:\n"
            "- Identify the smallest test-code change needed for the current failure or forced initial update.\n"
            f"- Use {update_evidence} as authoritative evidence about the contract change.\n"
            "- Preserve the original test structure, helper functions, style, assertions, and setup.\n"
            "- Patch only the lines that must change.\n\n"
            "OUTPUT STRICT JSON ONLY, with this schema:\n"
            "{\n"
            '  "patches": [\n'
            "    {\n"
            '      "start_line": 1,\n'
            '      "end_line": 1,\n'
            '      "replacement": "complete replacement text for that line range",\n'
            '      "reason": "why this localized change is needed"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- No markdown fences, no prose outside JSON.\n"
            "- Line numbers are 1-based and must refer to NUMBERED CURRENT TEST FILE.\n"
            "- For multi-line replacement, put newline characters inside the replacement string.\n"
            "- Do not output the complete test file unless the entire file is genuinely the line range being replaced."
        )
        return "\n\n".join(part for part in sections if part)

    def _build_static_review_patch_prompt(
        self,
        sources: dict[str, str],
        current_test: str,
        patch_history: list[dict[str, Any]],
    ) -> str:
        self._validate_ast_boundary(sources)
        update_evidence = self._contract_update_evidence()
        sections = []
        sections.append(
            "You are SCT-Agent with the Test Runner disabled for an ablation study.\n"
            "You must statically review the obsolete JS/TS test against the Solidity change "
            "and emit precise line-level JSON patches. Do not rewrite the whole test file."
        )

        if 3 in self.info:
            sections.append(self._section("CONTRACT DIFF Pold -> Pnew", sources.get("contract_diff", "")))
        if 9 in self.info:
            sections.append(
                self._section(
                    "PNEW TREE-SITTER AST / NEW SIGNATURES, TYPES, AND MODIFIERS",
                    sources.get("pnew_ast", ""),
                )
            )
        if 13 in self.info:
            sections.append(self._section("PATCH HISTORY", self._compact_patch_history(patch_history)))
        if 12 in self.info:
            sections.append(
                self._section("NUMBERED CURRENT TEST FILE", self.patcher.add_line_numbers(current_test))
            )

        sections.append(
            "TASK:\n"
            f"- Statically review whether the current test reflects {update_evidence}.\n"
            "- If a concrete static issue remains, output localized JSON line patches.\n"
            "- If no concrete issue is found, set pass=true and patches=[].\n"
            "- Do not use or invent runtime feedback.\n\n"
            "OUTPUT STRICT JSON ONLY, with this schema:\n"
            "{\n"
            '  "pass": false,\n'
            '  "issues": ["specific static issue, empty only when pass=true"],\n'
            '  "patches": [\n'
            "    {\n"
            '      "start_line": 1,\n'
            '      "end_line": 1,\n'
            '      "replacement": "complete replacement text for that line range",\n'
            '      "reason": "why this localized change is needed"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- No markdown fences, no prose outside JSON.\n"
            "- If no concrete static issue is found, output exactly "
            '{"pass": true, "issues": [], "patches": []}.\n'
            "- Line numbers are 1-based and must refer to NUMBERED CURRENT TEST FILE.\n"
            "- For multi-line replacement, put newline characters inside the replacement string.\n"
            "- Preserve the original test structure and style."
        )
        return "\n\n".join(part for part in sections if part)

    def _parse_static_review_response(self, response: str) -> dict[str, Any]:
        payload = self.patcher._extract_json(response)
        if not isinstance(payload, dict):
            raise ValueError("Static review response must be a JSON object.")

        patches = payload.get("patches", [])
        if not isinstance(patches, list):
            raise ValueError("'patches' must be a list.")

        normalized = []
        for idx, patch in enumerate(patches, start=1):
            if not isinstance(patch, dict):
                raise ValueError(f"Patch #{idx} is not an object.")
            normalized.append(
                {
                    "start_line": int(patch["start_line"]),
                    "end_line": int(patch["end_line"]),
                    "replacement": str(patch.get("replacement", "")),
                    "reason": str(patch.get("reason", "")),
                }
            )

        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]

        return {
            "pass": bool(payload.get("pass", False)),
            "issues": [str(issue) for issue in issues],
            "patches": normalized,
        }

    @staticmethod
    def _section(title: str, content: str) -> str:
        if not content:
            return f"=== {title} ===\n<empty>"
        return f"=== {title} ===\n{content}"

    @staticmethod
    def _compact_patch_history(patch_history: list[dict[str, Any]], limit: int = 5) -> str:
        if not patch_history:
            return "[]"
        compact = []
        for item in patch_history[-limit:]:
            compact.append(
                {
                    "iteration": item.get("iteration"),
                    "patches": item.get("patches"),
                    "syntax_ok": item.get("syntax_ok"),
                    "error_message": (item.get("judgement") or {}).get("message", ""),
                }
            )
        return json.dumps(compact, indent=2, ensure_ascii=False)
