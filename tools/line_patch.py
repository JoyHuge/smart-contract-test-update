"""
line_patch.py - structured line-level patch application for SCT-Agent.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PatchResult:
    ok: bool
    text: str
    message: str
    syntax_ok: bool = True
    syntax_log: str = ""


class LinePatchApplicator:
    """Apply LLM-produced JSON patches to an in-memory full test file."""

    def parse_patch_response(self, response: str) -> list[dict[str, Any]]:
        payload = self._extract_json(response)
        if isinstance(payload, list):
            patches = payload
        elif isinstance(payload, dict):
            patches = payload.get("patches", [])
        else:
            patches = []
        if not isinstance(patches, list):
            raise ValueError("Patch JSON must contain a list field named 'patches'.")
        parsed = []
        for idx, patch in enumerate(patches, start=1):
            if not isinstance(patch, dict):
                raise ValueError(f"Patch #{idx} is not an object.")
            start_line = int(patch["start_line"])
            end_line = int(patch["end_line"])
            replacement = str(patch.get("replacement", ""))
            reason = str(patch.get("reason", ""))
            parsed.append(
                {
                    "start_line": start_line,
                    "end_line": end_line,
                    "replacement": replacement,
                    "reason": reason,
                }
            )
        return parsed

    def apply(self, source: str, patches: list[dict[str, Any]], syntax_check: bool = True) -> PatchResult:
        if not patches:
            return PatchResult(False, source, "No patches were provided.")

        newline = "\r\n" if "\r\n" in source else "\n"
        has_trailing_newline = source.endswith(("\n", "\r\n"))
        lines = source.splitlines()
        original_count = len(lines)

        normalized = []
        for patch in patches:
            start_line = int(patch["start_line"])
            end_line = int(patch["end_line"])
            if start_line < 1 or end_line < start_line:
                return PatchResult(False, source, f"Invalid line range: {start_line}-{end_line}.")
            if end_line > original_count:
                return PatchResult(
                    False,
                    source,
                    f"Line range {start_line}-{end_line} exceeds file length {original_count}.",
                )
            replacement_lines = str(patch.get("replacement", "")).splitlines()
            normalized.append((start_line, end_line, replacement_lines))

        normalized.sort(key=lambda item: item[0])
        previous_end = 0
        for start_line, end_line, _ in normalized:
            if start_line <= previous_end:
                return PatchResult(False, source, "Overlapping patches are not allowed.")
            previous_end = end_line

        patched = list(lines)
        for start_line, end_line, replacement_lines in reversed(normalized):
            patched[start_line - 1 : end_line] = replacement_lines

        new_text = newline.join(patched)
        if has_trailing_newline:
            new_text += newline

        if syntax_check:
            syntax_ok, syntax_log = self._node_check(new_text)
            if not syntax_ok:
                return PatchResult(
                    True,
                    new_text,
                    "Patch applied, but JavaScript syntax check failed.",
                    syntax_ok=False,
                    syntax_log=syntax_log,
                )

        return PatchResult(True, new_text, "Patch applied successfully.")

    @staticmethod
    def add_line_numbers(source: str) -> str:
        lines = source.splitlines()
        width = max(4, len(str(len(lines))))
        return "\n".join(f"{idx:>{width}} | {line}" for idx, line in enumerate(lines, start=1))

    @staticmethod
    def _extract_json(response: str) -> Any:
        text = (response or "").strip()
        fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        else:
            start_obj = text.find("{")
            start_arr = text.find("[")
            starts = [pos for pos in (start_obj, start_arr) if pos >= 0]
            if starts:
                start = min(starts)
                end = max(text.rfind("}"), text.rfind("]"))
                if end >= start:
                    text = text[start : end + 1]
        return json.loads(text)

    @staticmethod
    def _node_check(source: str) -> tuple[bool, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0, (result.stdout or "") + (result.stderr or "")
        except Exception as exc:
            return False, str(exc)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
