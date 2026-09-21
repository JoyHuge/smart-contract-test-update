from pathlib import Path
from typing import List, Optional
import os


class TestFileManager:
    """Manage generated test files with model-specific suffixes."""

    def __init__(self, project_root: str, dataset_project_prefix: str = ""):
        self.project_root = Path(project_root)
        self.dataset_project_prefix = (dataset_project_prefix or "").strip()

    def _suffix_match_relative_path(self, path: Path) -> Optional[Path]:
        parts = path.parts
        project_parts = self.project_root.parts
        max_prefix_len = min(len(parts), len(project_parts))
        for prefix_len in range(max_prefix_len, 0, -1):
            if tuple(parts[:prefix_len]) == tuple(project_parts[-prefix_len:]):
                return Path(*parts[prefix_len:])
        return None

    @staticmethod
    def _path_exists_under_root(project_root: Path, relative: Path) -> bool:
        target = project_root / relative
        return target.exists() or target.parent.exists()

    def _repo_relative_path(self, original_path: str) -> Path:
        path = Path(original_path)
        parts = path.parts
        if self.dataset_project_prefix and parts and parts[0] == self.dataset_project_prefix:
            return Path(*parts[1:])
        return path

    def _resolve_via_repo_layout(self, repo_relative: Path) -> Optional[Path]:
        """Locate repo-relative test path when project_root is a monorepo sub-package."""
        resolved_root = self.project_root.resolve()
        current = resolved_root
        for _ in range(12):
            target = current / repo_relative
            if target.exists() or (repo_relative.parts and target.parent.exists()):
                try:
                    return target.relative_to(resolved_root)
                except ValueError:
                    return Path(os.path.relpath(str(target), str(resolved_root)))
            if current.parent == current:
                break
            current = current.parent
        return None

    def _project_relative_path(self, original_path: str) -> Path:
        path = Path(original_path)
        if path.is_absolute():
            try:
                return path.relative_to(self.project_root)
            except ValueError:
                pass

        repo_relative = self._repo_relative_path(original_path)
        nested = self._suffix_match_relative_path(repo_relative)
        if nested is not None and (self.project_root / nested).exists():
            return nested

        via_layout = self._resolve_via_repo_layout(repo_relative)
        if via_layout is not None:
            return via_layout

        parts = path.parts
        candidates: List[Path] = []

        suffix_match = self._suffix_match_relative_path(path)
        if suffix_match is not None:
            candidates.append(suffix_match)

        if self.dataset_project_prefix and parts and parts[0] == self.dataset_project_prefix:
            candidates.append(Path(*parts[1:]))

        for i in range(len(parts)):
            candidates.append(Path(*parts[i:]))

        seen = set()
        valid: List[Path] = []
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if self._path_exists_under_root(self.project_root, candidate):
                valid.append(candidate)

        if valid:
            if (
                self.dataset_project_prefix
                and parts
                and parts[0] == self.dataset_project_prefix
            ):
                stripped = Path(*parts[1:])
                if stripped in valid:
                    return stripped

            existing = [c for c in valid if (self.project_root / c).exists()]
            pool = existing or valid

            def score(relative: Path) -> tuple:
                head_penalty = 0
                if (
                    self.dataset_project_prefix
                    and relative.parts
                    and relative.parts[0] == self.dataset_project_prefix
                ):
                    head_penalty = 10
                return (head_penalty, len(relative.parts))

            return min(pool, key=score)

        if self.dataset_project_prefix and parts and parts[0] == self.dataset_project_prefix:
            stripped = Path(*parts[1:])
            fallback = self._resolve_via_repo_layout(stripped)
            if fallback is not None:
                return fallback
            return stripped

        if suffix_match is not None:
            return suffix_match

        fallback = self._resolve_via_repo_layout(repo_relative)
        if fallback is not None:
            return fallback
        return path

    def _legacy_prefixed_path(self, original_path: str, model_name: str = "llm") -> Optional[Path]:
        """Return legacy nested path project_root/<prefix>/... if an old save exists."""
        parts = Path(original_path).parts
        if not self.dataset_project_prefix or not parts:
            return None
        if parts[0] != self.dataset_project_prefix:
            return None
        legacy = self.project_root / Path(*parts)
        legacy_llm = legacy.with_name(f"{legacy.stem}_{model_name}{legacy.suffix}")
        return legacy_llm if legacy_llm.exists() else None

    def get_llm_test_path(self, original_path: str, model_name: str = "llm") -> Path:
        """Return the path for a generated test with a model suffix."""
        path = self.project_root / self._project_relative_path(original_path)
        stem = path.stem
        suffix = path.suffix
        llm_path = path.with_name(f"{stem}_{model_name}{suffix}")
        return llm_path

    def save_llm_test(
        self,
        original_path: str,
        content: str,
        overwrite: bool = False,
        model_name: str = "llm"
    ) -> str:
        """Save an LLM-generated test file."""
        llm_path = self.get_llm_test_path(original_path, model_name=model_name)

        if llm_path.exists() and not overwrite:
            raise FileExistsError(
                f"LLM test file already exists: {llm_path}"
            )

        llm_path.parent.mkdir(parents=True, exist_ok=True)

        with open(llm_path, 'w', encoding='utf-8') as f:
            f.write(content)

        legacy_path = self._legacy_prefixed_path(original_path, model_name=model_name)
        if legacy_path and legacy_path != llm_path and legacy_path.exists():
            legacy_path.unlink()

        return str(llm_path.relative_to(self.project_root))

    def get_llm_test(self, original_path: str, model_name: str = "llm") -> Optional[str]:
        """Read an LLM-generated test file."""
        llm_path = self.get_llm_test_path(original_path, model_name=model_name)
        if llm_path.exists():
            with open(llm_path, 'r', encoding='utf-8') as f:
                return f.read()

        legacy_path = self._legacy_prefixed_path(original_path, model_name=model_name)
        if legacy_path is None:
            return None

        with open(legacy_path, 'r', encoding='utf-8') as f:
            content = f.read()
        llm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(llm_path, 'w', encoding='utf-8') as f:
            f.write(content)
        legacy_path.unlink(missing_ok=True)
        return content

    def _normalize_path(self, original_path: str) -> Path:
        """Normalize a dataset path and remove duplicate project prefixes."""
        return self.project_root / self._project_relative_path(original_path)

    def get_original_test(self, original_path: str) -> Optional[str]:
        """Read the original developer-written ground-truth test."""
        original = self._normalize_path(original_path)
        if original.exists():
            with open(original, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def delete_llm_test(self, original_path: str, model_name: str = "llm") -> bool:
        """Delete an LLM-generated test file."""
        llm_path = self.get_llm_test_path(original_path, model_name=model_name)
        if llm_path.exists():
            llm_path.unlink()
            return True
        return False

    def llm_test_exists(self, original_path: str, model_name: str = "llm") -> bool:
        """Return whether a generated test file exists."""
        return self.get_llm_test_path(original_path, model_name=model_name).exists()

    def list_all_llm_tests(self) -> List[str]:
        """List all generated test files."""
        llm_files = []
        base_suffixes = ("deepseek", "glm", "gpt", "claude", "gemini", "qwen", "llm")
        mode_suffixes = (
            "full",
            "wo_test_runner",
            "wo_ast_wo_test_runner",
        )
        model_suffixes = base_suffixes + tuple(
            f"{model}_{mode}" for model in base_suffixes for mode in mode_suffixes
        )
        for file_path in self.project_root.rglob("*.*"):
            if "test" in file_path.parts:
                stem = file_path.stem
                if any(stem.endswith(f"_{suffix}") for suffix in model_suffixes):
                    relative_path = str(file_path.relative_to(self.project_root))
                    llm_files.append(relative_path)
        return llm_files

    def cleanup_all_llm_tests(self) -> int:
        """Delete all generated test files."""
        llm_files = self.list_all_llm_tests()
        for file_path in llm_files:
            self.delete_llm_test(file_path)
        return len(llm_files)
