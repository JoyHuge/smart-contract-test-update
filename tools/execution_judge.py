"""
execution_judge.py - compact execution feedback extraction for SCT-Agent.
"""

from __future__ import annotations

import json
import re
from typing import Any


class ExecutionJudge:
    """Turn raw sandbox output into dense feedback for the planner."""

    def judge(self, success: bool, raw_log: str, framework: str = "truffle") -> dict[str, Any]:
        clean_log = self._clean(raw_log)
        return {
            "status": "pass" if success else "fail",
            "phase": self._phase(success, clean_log, framework),
            "framework": framework,
            "failing_tests": self._failing_tests(clean_log),
            "error_type": self._error_type(clean_log),
            "message": self._message(clean_log),
            "revert_reason": self._revert_reason(clean_log),
            "suspected_lines": self._suspected_lines(clean_log),
            "stack_excerpt": self._excerpt(clean_log),
        }

    def to_prompt_text(self, judged: dict[str, Any]) -> str:
        return json.dumps(judged, indent=2, ensure_ascii=False)

    @staticmethod
    def _clean(log: str) -> str:
        log = re.sub(r"\x1b\[[0-9;]*m", "", log or "")
        lines = [line.rstrip() for line in log.splitlines()]
        filtered = []
        noisy = (
            "Browserslist:",
            "Warning:",
            "DeprecationWarning",
            "npm WARN",
            "yarn run",
            "$ ",
        )
        for line in lines:
            if any(line.strip().startswith(prefix) for prefix in noisy):
                continue
            filtered.append(line)
        return "\n".join(filtered).strip()

    @staticmethod
    def _phase(success: bool, log: str, framework: str) -> str:
        if success:
            return "pass"
        lowered = log.lower()
        if "syntaxerror" in lowered or "unexpected token" in lowered:
            return "syntax"
        if "compilation failed" in lowered or "compileerror" in lowered or "parsererror" in lowered:
            return "compile"
        if "error: cannot find module" in lowered or "cannot find module" in lowered:
            return "environment"
        return "test"

    @staticmethod
    def _failing_tests(log: str) -> list[str]:
        names = []
        for match in re.finditer(r"^\s*\d+\)\s+(.+)$", log, re.MULTILINE):
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)
        return names[:8]

    @staticmethod
    def _error_type(log: str) -> str:
        patterns = [
            r"\b(AssertionError)\b",
            r"\b(TypeError)\b",
            r"\b(ReferenceError)\b",
            r"\b(SyntaxError)\b",
            r"\b(ParserError)\b",
            r"\b(CompileError)\b",
            r"\b(Error):",
        ]
        for pattern in patterns:
            match = re.search(pattern, log)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _message(log: str) -> str:
        candidates = []
        for pattern in [
            r"(TypeError: .+)",
            r"(AssertionError: .+)",
            r"(ReferenceError: .+)",
            r"(SyntaxError: .+)",
            r"(Error: .+)",
            r"(VM Exception while processing transaction: .+)",
            r"(Returned error: .+)",
        ]:
            for match in re.finditer(pattern, log):
                candidates.append(match.group(1).strip())
        if candidates:
            return candidates[0][:1000]
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return "\n".join(lines[-8:])[:1000]

    @staticmethod
    def _revert_reason(log: str) -> str:
        patterns = [
            r"revert(?:ed)?(?: with reason string)? ['\"]([^'\"]+)['\"]",
            r"reason(?: string)?:\s*['\"]?([^'\"]+)",
            r"VM Exception while processing transaction: revert\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, log, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:500]
        return ""

    @staticmethod
    def _suspected_lines(log: str) -> list[int]:
        lines = []
        for match in re.finditer(r":(\d+):\d+\)?", log):
            value = int(match.group(1))
            if value not in lines:
                lines.append(value)
        return lines[:12]

    @staticmethod
    def _excerpt(log: str, max_lines: int = 80) -> str:
        lines = [line for line in log.splitlines() if line.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        head = lines[:20]
        tail = lines[-60:]
        return "\n".join(head + ["... (middle omitted) ..."] + tail)
