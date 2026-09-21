from __future__ import annotations

import json
import os
import shlex
from pathlib import Path


class TestCommandRegistry:
    def __init__(self, config_path: str | None = None):
        configured = config_path or os.environ.get("SCT_TEST_COMMANDS_PATH")
        if configured:
            self.config_path = Path(configured)
        else:
            self.config_path = Path(__file__).resolve().parents[1] / "test_commands.json"
        self._entries: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._entries is not None:
            return self._entries
        if not self.config_path.exists():
            self._entries = []
            return self._entries
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        self._entries = data if isinstance(data, list) else []
        return self._entries

    @staticmethod
    def _record_index_matches(entry: dict, record_index: int) -> bool:
        value = entry.get("record_index")
        if isinstance(value, list):
            return record_index in value
        return value == record_index

    def find(self, record_index: int | None = None, test_file: str = "", project_path: str = "") -> dict | None:
        entries = self._load()
        if record_index is not None:
            for entry in entries:
                if self._record_index_matches(entry, record_index):
                    return entry

        if project_path:
            path_parts = Path(project_path).parts
            for entry in entries:
                project = entry.get("project")
                if project and project in path_parts:
                    return entry
        return None


def format_command(command_template: str, test_file: str) -> str:
    if "{test_file}" not in command_template:
        raise ValueError(
            "Test command template must include {test_file}; "
            f"got: {command_template}"
        )
    quoted_test_file = shlex.quote(test_file)
    return command_template.replace("{test_file}", quoted_test_file)
