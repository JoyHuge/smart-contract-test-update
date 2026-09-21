from __future__ import annotations

import os
import json
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path


_SECTION_RE = re.compile(r"^## (?P<project>.+?)\s*$", re.MULTILINE)
_BASH_BLOCK_RE = re.compile(r"```bash\n(?P<body>.*?)\n```", re.DOTALL)
_FIELD_RE = re.compile(r"^- (?P<key>[a-zA-Z_]+): `(?P<value>.*?)`\s*$", re.MULTILINE)
_CHECKOUT_RE = re.compile(r"\bgit\s+checkout\s+-f\s+[0-9a-fA-F]{7,40}\b")
_PID_ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)=\$!\s*$")
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?")
_NVM_USE_RE = re.compile(r"^\s*nvm\s+use(?:\s+(?P<version>\S+))?\s*$")


def _normalize_record_indices(data: dict) -> frozenset[int]:
    indices: set[int] = set()
    raw_indices = data.get("record_indices")
    if raw_indices is not None:
        if isinstance(raw_indices, str):
            raw_indices = [part.strip() for part in raw_indices.split(",") if part.strip()]
        indices.update(int(value) for value in raw_indices)

    record_index = data.get("record_index")
    if record_index is not None:
        if isinstance(record_index, list):
            indices.update(int(value) for value in record_index)
        else:
            indices.add(int(record_index))

    return frozenset(indices)


def _validate_recipes(recipes: list["ReproRecipe"]) -> None:
    index_to_label: dict[int, str] = {}
    project_default_counts: dict[str, int] = {}

    for recipe in recipes:
        label = f"{recipe.project} ({sorted(recipe.record_indices)})"
        for record_index in recipe.record_indices:
            if record_index in index_to_label:
                raise ValueError(
                    f"Duplicate record_index {record_index} in repro recipes: "
                    f"{index_to_label[record_index]} and {label}"
                )
            index_to_label[record_index] = label

        if recipe.default_for_project:
            project_default_counts[recipe.project] = project_default_counts.get(recipe.project, 0) + 1

    for project, count in project_default_counts.items():
        if count > 1:
            raise ValueError(f"Multiple default_for_project repro recipes for project {project!r}")


@dataclass(frozen=True)
class ReproRecipe:
    project: str
    record_indices: frozenset[int]
    framework: str
    commit: str
    original_test: str
    cwd: str
    setup_commands: tuple[str, ...]
    test_command: str
    cleanup_commands: tuple[str, ...]
    pre_test_commands: tuple[str, ...] = ()
    default_for_project: bool = False

    @property
    def record_index(self) -> int | None:
        if len(self.record_indices) == 1:
            return next(iter(self.record_indices))
        return None

    def to_json_dict(self) -> dict:
        payload = {
            "project": self.project,
            "record_indices": sorted(self.record_indices),
            "framework": self.framework,
            "commit": self.commit,
            "original_test": self.original_test,
            "cwd": self.cwd,
            "setup_commands": list(self.setup_commands),
            "test_command": self.test_command,
            "cleanup_commands": list(self.cleanup_commands),
        }
        if self.pre_test_commands:
            payload["pre_test_commands"] = list(self.pre_test_commands)
        if self.default_for_project:
            payload["default_for_project"] = True
        return payload

    @classmethod
    def from_json_dict(cls, data: dict) -> "ReproRecipe":
        return cls(
            project=str(data.get("project", "")),
            record_indices=_normalize_record_indices(data),
            framework=str(data.get("framework", "")),
            commit=str(data.get("commit", "")),
            original_test=str(data.get("original_test", data.get("test", ""))),
            cwd=str(data.get("cwd", "")),
            setup_commands=tuple(data.get("setup_commands") or []),
            test_command=str(data.get("test_command", "")),
            cleanup_commands=tuple(data.get("cleanup_commands") or []),
            pre_test_commands=tuple(data.get("pre_test_commands") or []),
            default_for_project=bool(data.get("default_for_project", False)),
        )


def _default_repro_commands_path() -> Path:
    script_dir = Path(__file__).resolve().parents[1]
    json_path = script_dir / "repro_commands.json"
    if json_path.exists():
        return json_path
    return script_dir / "repro_commands.md"


def _strip_project_prefix(project: str, path: str) -> str:
    parts = Path(path).parts
    if parts and parts[0] == project:
        return str(Path(*parts[1:]))
    return path


def _cwd_relative_to_project(recipe: ReproRecipe) -> Path:
    cwd_parts = Path(recipe.cwd).parts
    if recipe.project not in cwd_parts:
        return Path()
    project_index = len(cwd_parts) - 1 - list(reversed(cwd_parts)).index(recipe.project)
    tail = cwd_parts[project_index + 1 :]
    return Path(*tail) if tail else Path()


def _test_path_for_recipe_cwd(recipe: ReproRecipe, test_file: str) -> str:
    stripped_test = Path(_strip_project_prefix(recipe.project, test_file))
    cwd_relative = _cwd_relative_to_project(recipe)
    if not cwd_relative.parts:
        return str(stripped_test)
    if stripped_test.parts and stripped_test.parts[0] == "..":
        return str(stripped_test)
    try:
        return str(stripped_test.relative_to(cwd_relative))
    except ValueError:
        pass

    cwd = Path(recipe.cwd)
    if (cwd / stripped_test).exists() or (cwd / stripped_test).parent.exists():
        return str(stripped_test)

    up = Path(*([".."] * len(cwd_relative.parts)))
    return str(up / stripped_test)


def _extract_cwd(commands: list[str]) -> str:
    cwd: Path | None = None
    for command in commands:
        stripped = command.strip()
        if stripped.startswith("cd "):
            try:
                target = shlex.split(stripped, posix=True)[1]
            except Exception:
                target = stripped[3:].strip()
            target_path = Path(target).expanduser()
            if target_path.is_absolute() or cwd is None:
                cwd = target_path
            else:
                cwd = (cwd / target_path)
            cwd = cwd.resolve(strict=False)
    return str(cwd) if cwd is not None else ""


def _split_logical_commands(script: str) -> list[str]:
    commands: list[str] = []
    current: list[str] = []
    heredoc_tag: str | None = None

    for raw_line in script.splitlines():
        line = raw_line.rstrip()

        if heredoc_tag:
            current.append(line)
            if line.strip() == heredoc_tag:
                commands.append("\n".join(current).strip())
                current = []
                heredoc_tag = None
            continue

        if not current and not line.strip():
            continue

        current.append(line)
        heredoc_match = _HEREDOC_RE.search(line)
        if heredoc_match:
            heredoc_tag = heredoc_match.group("tag")
            continue

        if line.endswith("\\"):
            continue

        commands.append("\n".join(current).strip())
        current = []

    if current:
        commands.append("\n".join(current).strip())

    return [command for command in commands if command]


def _metadata_from_section(section: str) -> dict[str, str]:
    fields = {m.group("key"): m.group("value") for m in _FIELD_RE.finditer(section)}
    return fields


def _has_test_command_marker(command: str) -> bool:
    compact = " ".join(command.split())
    return bool(
        re.search(r"\b(truffle|hardhat|buidler)\b.*\btest\b", compact)
        or re.search(r"\bforge\b.*\b(match-path|test)\b", compact)
        or re.search(r"\bmocha\b", compact)
        or "scripts/test.sh" in compact
        or "TRUFFLE_TEST=true" in compact
        or "process.argv=" in compact
    )


def _looks_like_test_command(command: str, original_test: str, original_relative_test: str) -> bool:
    compact = " ".join(command.split())
    if not compact or compact.startswith("#"):
        return False

    references_original = (
        original_test and original_test in command
    ) or (
        original_relative_test and original_relative_test in command
    ) or Path(original_test).name in command

    if not references_original:
        return False

    return _has_test_command_marker(command)


def _find_test_command_index(
    commands: list[str],
    original_test: str,
    original_relative_test: str,
) -> int | None:
    for index in range(len(commands) - 1, -1, -1):
        if _looks_like_test_command(commands[index], original_test, original_relative_test):
            return index
    for index in range(len(commands) - 1, -1, -1):
        if _has_test_command_marker(commands[index]):
            return index
    return None


def _is_transient_status_command(command: str) -> bool:
    stripped = command.strip()
    return bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\$\?", stripped)
        or re.fullmatch(r"exit\s+\$[A-Za-z_][A-Za-z0-9_]*", stripped)
    )


def _render_test_path(command: str, recipe: ReproRecipe, test_file: str) -> str:
    target_test = _test_path_for_recipe_cwd(recipe, test_file)
    rendered = command.replace("{test_file}", target_test)
    replacements = []
    if recipe.original_test:
        replacements.append(recipe.original_test)
        replacements.append(_strip_project_prefix(recipe.project, recipe.original_test))

    replaced = False
    for old in dict.fromkeys(value for value in replacements if value):
        if old in rendered:
            rendered = rendered.replace(old, target_test)
            replaced = True
    if not replaced and recipe.original_test:
        basename = re.escape(Path(recipe.original_test).name)
        rendered = re.sub(rf"[\w./-]*{basename}", target_test, rendered)
    return rendered


def _last_nvm_use_command(recipe: ReproRecipe) -> str:
    last = ""
    for command in recipe.setup_commands:
        stripped = command.strip()
        if _NVM_USE_RE.match(stripped):
            last = stripped
    return last


def _project_root_from_recipe(recipe: ReproRecipe) -> Path | None:
    """Return the project root embedded in a recipe's original cwd."""
    cwd_parts = Path(recipe.cwd).parts
    if not cwd_parts or recipe.project not in cwd_parts:
        return None
    project_index = len(cwd_parts) - 1 - list(reversed(cwd_parts)).index(recipe.project)
    return Path(*cwd_parts[: project_index + 1])


def _bind_recipe_to_project_path(
    recipe: ReproRecipe,
    project_path: str,
) -> ReproRecipe:
    """Rebase stored absolute paths onto the active project checkout.

    A stored recipe may have been created in a different environment. The
    caller-provided project path is authoritative at runtime.
    """
    if not project_path:
        return recipe

    # Worktree recipes deliberately execute outside the repository supplied by
    # the caller.  Rebinding their paths would turn the worktree-management
    # commands into commands against the main checkout (including a fatal
    # ``git worktree remove <main-checkout>``).
    if "sct-worktrees" in Path(recipe.cwd).parts:
        return recipe

    source_root = _project_root_from_recipe(recipe)
    if source_root is None:
        return recipe

    target_root = Path(project_path).expanduser().resolve(strict=False)
    source_root_text = str(source_root)
    target_root_text = str(target_root)
    if source_root_text == target_root_text:
        return recipe

    cwd_relative = _cwd_relative_to_project(recipe)
    target_cwd = target_root / cwd_relative

    def rebase(command: str) -> str:
        return command.replace(source_root_text, target_root_text)

    return replace(
        recipe,
        cwd=str(target_cwd),
        setup_commands=tuple(rebase(command) for command in recipe.setup_commands),
        test_command=rebase(recipe.test_command),
        cleanup_commands=tuple(rebase(command) for command in recipe.cleanup_commands),
        pre_test_commands=tuple(rebase(command) for command in recipe.pre_test_commands),
    )


class ReproCommandRegistry:
    def __init__(self, config_path: str | None = None):
        configured = config_path or os.environ.get("SCT_REPRO_COMMANDS_PATH")
        self.config_path = Path(configured) if configured else _default_repro_commands_path()
        self._recipes: list[ReproRecipe] | None = None

    def _load(self) -> list[ReproRecipe]:
        if self._recipes is not None:
            return self._recipes
        if not self.config_path.exists():
            self._recipes = []
            return self._recipes

        if self.config_path.suffix.lower() == ".json":
            self._recipes = self._load_json()
            return self._recipes

        self._recipes = self._load_markdown()
        return self._recipes

    def _load_json(self) -> list[ReproRecipe]:
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        entries = data if isinstance(data, list) else data.get("recipes", [])
        recipes = []
        for entry in entries:
            if isinstance(entry, dict):
                recipe = ReproRecipe.from_json_dict(entry)
                if recipe.project and recipe.test_command:
                    recipes.append(recipe)
        _validate_recipes(recipes)
        return recipes

    def _load_markdown(self) -> list[ReproRecipe]:
        text = self.config_path.read_text(encoding="utf-8")
        matches = list(_SECTION_RE.finditer(text))
        recipes: list[ReproRecipe] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section = text[start:end]
            block_match = _BASH_BLOCK_RE.search(section)
            if not block_match:
                continue

            project = match.group("project").strip()
            fields = _metadata_from_section(section)
            commands = _split_logical_commands(block_match.group("body"))
            original_test = fields.get("test", "")
            original_relative_test = _strip_project_prefix(project, original_test)
            test_command_index = _find_test_command_index(
                commands,
                original_test=original_test,
                original_relative_test=original_relative_test,
            )
            if test_command_index is None:
                continue

            cleanup_commands = tuple(
                command
                for command in commands[test_command_index + 1 :]
                if not _is_transient_status_command(command)
            )
            record_indices = _normalize_record_indices(fields)

            recipes.append(
                ReproRecipe(
                    project=project,
                    record_indices=record_indices,
                    framework=fields.get("framework", ""),
                    commit=fields.get("commit", ""),
                    original_test=original_test,
                    cwd=_extract_cwd(commands),
                    setup_commands=tuple(commands[:test_command_index]),
                    test_command=commands[test_command_index],
                    cleanup_commands=cleanup_commands,
                    default_for_project=str(fields.get("default_for_project", "")).lower() == "true",
                )
            )

        _validate_recipes(recipes)
        return recipes

    @staticmethod
    def _project_from_path(project_path: str) -> str:
        parts = Path(project_path).parts
        return parts[0] if parts else ""

    def find(
        self,
        record_index: int | None = None,
        project: str = "",
        project_path: str = "",
    ) -> ReproRecipe | None:
        recipes = self._load()
        project_name = project or self._project_from_path(project_path)
        selected: ReproRecipe | None = None

        if record_index is not None:
            matches = [recipe for recipe in recipes if record_index in recipe.record_indices]
            if len(matches) == 1:
                selected = matches[0]
            elif len(matches) > 1:
                labels = ", ".join(
                    f"{recipe.project} ({sorted(recipe.record_indices)})" for recipe in matches
                )
                raise ValueError(
                    f"Ambiguous repro recipe for record_index={record_index}: {labels}"
                )
            elif project_name:
                defaults = [
                    recipe
                    for recipe in recipes
                    if recipe.project == project_name and recipe.default_for_project
                ]
                if len(defaults) == 1:
                    selected = defaults[0]
        elif project_name:
            defaults = [
                recipe
                for recipe in recipes
                if recipe.project == project_name and recipe.default_for_project
            ]
            if len(defaults) == 1:
                selected = defaults[0]

        if selected is None:
            return None
        return _bind_recipe_to_project_path(selected, project_path)

    def render_setup_script(
        self,
        recipe: ReproRecipe,
        commit_sha: str,
        pid_dir: Path,
    ) -> str:
        pid_dir = pid_dir.resolve()
        lines = [
            "set -e",
            f"mkdir -p {shlex.quote(str(pid_dir))}",
            # Keep transient or corrupt package downloads out of shared user
            # caches.  The cache is discarded before the next setup attempt
            # for this record, while package lockfiles still define versions.
            f"export npm_config_cache={shlex.quote(str(pid_dir / 'npm-cache'))}",
            f"export YARN_CACHE_FOLDER={shlex.quote(str(pid_dir / 'yarn-cache'))}",
            # Yarn classic honors the absolute `resolved` URLs in old lockfiles,
            # even when the command includes --registry.  Those URLs currently
            # point at registry.yarnpkg.com, whose tarball responses have been
            # intermittently truncated in this environment.  Rewrite only the
            # interchangeable registry host just before every Yarn installation;
            # package names, versions, hashes, and all Git dependencies remain
            # untouched.
            "export SCT_PACKAGE_REGISTRY=${SCT_PACKAGE_REGISTRY:-https://registry.npmjs.org}",
            "export npm_config_registry=$SCT_PACKAGE_REGISTRY",
            "export npm_config_fetch_retries=${SCT_NPM_FETCH_RETRIES:-5}",
            "export npm_config_fetch_retry_mintimeout=${SCT_NPM_FETCH_RETRY_MINTIMEOUT:-2000}",
            "export npm_config_fetch_retry_maxtimeout=${SCT_NPM_FETCH_RETRY_MAXTIMEOUT:-30000}",
            "export YARN_NETWORK_TIMEOUT=${SCT_YARN_NETWORK_TIMEOUT:-600000}",
            "export YARN_NETWORK_CONCURRENCY=${SCT_YARN_NETWORK_CONCURRENCY:-1}",
            "export SCT_REPRO_INSTALL_ATTEMPTS=${SCT_REPRO_INSTALL_ATTEMPTS:-3}",
            "sct_prepare_yarn_lock() {",
            "  [ -f yarn.lock ] || return 0",
            "  python3 - \"$SCT_PACKAGE_REGISTRY\" <<'PY'",
            "from pathlib import Path",
            "import sys",
            "lock = Path('yarn.lock')",
            "text = lock.read_text()",
            "replacement = sys.argv[1].rstrip('/') + '/'",
            "updated = text.replace('https://registry.yarnpkg.com/', replacement)",
            "if updated != text:",
            "    lock.write_text(updated)",
            "PY",
            "}",
            "yarn() { if [ \"$1\" = install ]; then local attempt; sct_prepare_yarn_lock; for attempt in $(seq 1 \"$SCT_REPRO_INSTALL_ATTEMPTS\"); do if command yarn \"$@\" --registry \"$SCT_PACKAGE_REGISTRY\" --network-timeout \"$YARN_NETWORK_TIMEOUT\" --network-concurrency \"$YARN_NETWORK_CONCURRENCY\"; then return 0; fi; rm -rf \"$YARN_CACHE_FOLDER\"; [ \"$attempt\" -eq \"$SCT_REPRO_INSTALL_ATTEMPTS\" ] && return 1; sleep $((attempt * 5)); done; else command yarn \"$@\"; fi; }",
            "npm() { if [ \"$1\" = install ] || [ \"$1\" = ci ]; then local attempt; for attempt in $(seq 1 \"$SCT_REPRO_INSTALL_ATTEMPTS\"); do if command npm \"$@\" --registry \"$SCT_PACKAGE_REGISTRY\"; then return 0; fi; rm -rf \"$npm_config_cache\"; [ \"$attempt\" -eq \"$SCT_REPRO_INSTALL_ATTEMPTS\" ] && return 1; sleep $((attempt * 5)); done; else command npm \"$@\"; fi; }",
            # Several legacy recipes obtain Yarn with `npx -p yarn`.  That
            # bypasses the Yarn shell function above, so prepare its lockfile
            # at the npx boundary as well.
            "npx() { local arg; for arg in \"$@\"; do case \"$arg\" in yarn|yarn@*|*/yarn|*/yarn@*) sct_prepare_yarn_lock; break ;; esac; done; command npx \"$@\"; }",
            "export -f sct_prepare_yarn_lock yarn npm npx",
        ]
        for command in recipe.setup_commands:
            rendered = _CHECKOUT_RE.sub(f"git checkout -f {commit_sha}", command)
            lines.append(rendered)
            pid_match = _PID_ASSIGN_RE.match(rendered.strip())
            if pid_match:
                name = pid_match.group("name")
                lines.append(f"printf '%s\\n' \"${name}\" > {shlex.quote(str(pid_dir / f'{name}.pid'))}")
        return "\n".join(lines) + "\n"

    def render_cleanup_script(self, recipe: ReproRecipe, pid_dir: Path) -> str:
        pid_dir = pid_dir.resolve()
        lines = ["set +e"]
        if recipe.cwd:
            lines.append(f"cd {shlex.quote(recipe.cwd)}")
        for pid_path in sorted(pid_dir.glob("*.pid")):
            name = pid_path.stem
            lines.append(f"{name}=\"$(cat {shlex.quote(str(pid_path))} 2>/dev/null || true)\"")
        lines.extend(recipe.cleanup_commands)
        return "\n".join(lines) + "\n"

    def format_test_command(self, recipe: ReproRecipe, test_file: str) -> str:
        rendered = _render_test_path(recipe.test_command, recipe, test_file)
        nvm_use = _last_nvm_use_command(recipe)
        if nvm_use and "nvm use" not in rendered:
            rendered = f"source ~/.nvm/nvm.sh && {nvm_use} >/dev/null && {rendered}"
        return rendered

    @staticmethod
    def is_non_original_test_file(recipe: ReproRecipe, test_file: str) -> bool:
        original = Path(_strip_project_prefix(recipe.project, recipe.original_test)).name
        candidate = Path(_strip_project_prefix(recipe.project, test_file)).name
        return candidate != original

    def render_pre_test_script(self, recipe: ReproRecipe) -> str:
        if not recipe.pre_test_commands:
            return ""
        lines = ["set -e"]
        if recipe.cwd:
            lines.append(f"cd {shlex.quote(recipe.cwd)}")
        nvm_use = _last_nvm_use_command(recipe)
        if nvm_use:
            lines.append("source ~/.nvm/nvm.sh")
            lines.append(nvm_use)
        lines.extend(recipe.pre_test_commands)
        return "\n".join(lines) + "\n"


def export_markdown_recipes_to_json(markdown_path: Path, json_path: Path) -> int:
    registry = ReproCommandRegistry(str(markdown_path))
    recipes = registry._load()
    json_path.write_text(
        json.dumps([recipe.to_json_dict() for recipe in recipes], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return len(recipes)
