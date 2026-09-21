"""
Build file-level smart-contract production-test co-evolution samples.

This module:
- emits complete files rather than function snippets;
- applies strict evidence that a test change follows a contract change; and
- retains only high-confidence causal samples.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from .config import DEBUG, MAX_LOOKAHEAD_COMMITS, OUTPUT_DIR, REPO_ROOT, TARGET_REPOS
    from .repo_scan import collect_files, project_eligible
    from .link_tests import build_link_map, _ts_extract_contract_names
    from .git_miner import CommitChange, collect_commit_changes
    from .utils import normalize_repo_url
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from reconstruct_dataset.config import DEBUG, MAX_LOOKAHEAD_COMMITS, OUTPUT_DIR, REPO_ROOT, TARGET_REPOS
    from reconstruct_dataset.repo_scan import collect_files, project_eligible
    from reconstruct_dataset.link_tests import build_link_map, _ts_extract_contract_names
    from reconstruct_dataset.git_miner import CommitChange, collect_commit_changes
    from reconstruct_dataset.utils import normalize_repo_url

try:
    import git as gitpython
except Exception:
    gitpython = None

# ---------------------------------------------------------------------------
# File-level configuration
# ---------------------------------------------------------------------------
FILELEVEL_LOOKAHEAD = 30  # Delayed-commit search window.


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log_print(message: str, log_file=None):
    print(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def generate_output_filename(output_dir: Path) -> Tuple[str, str]:
    date_str = datetime.now().strftime("%y%m%d")
    base_name = f"total_data_filelevel_{date_str}"
    existing = list(output_dir.glob(f"{base_name}_*.json"))
    if existing:
        numbers = []
        for f in existing:
            parts = f.stem.split("_")
            if len(parts) >= 3 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1
    suffix = f"{next_num:02d}"
    return f"{base_name}_{suffix}.json", f"{base_name}_{suffix}.log"


def _is_actual_test_file(file_path: Path, content: str) -> bool:
    """Return whether a file appears to test a smart contract.
    Requirements: 1) test framework syntax (contract/describe/it);
              2) smart-contract interaction evidence.
    """
    if not content:
        return False

    # Condition 1: test framework syntax.
    has_test_block = (
        bool(re.search(r"\bcontract\s*\(\s*['\"]", content)) or
        bool(re.search(r"\bdescribe\s*\(\s*['\"]", content))
    )
    if not has_test_block:
        return False

    # Condition 2: smart-contract interaction evidence.
    has_contract_interaction = (
        bool(re.search(r"artifacts\.require\s*\(", content)) or           # Truffle style.
        bool(re.search(r"\b[A-Z][A-Za-z0-9]*\s*\.\s*new\s*\(", content)) or  # Contract.new()
        bool(re.search(r"\b[A-Z][A-Za-z0-9]*\s*\.\s*deployed\s*\(", content)) or  # Contract.deployed()
        bool(re.search(r"\b[A-Z][A-Za-z0-9]*\s*\.\s*at\s*\(", content)) or      # Contract.at()
        bool(re.search(r"\bContract\b.*\.new\b", content))                    # Generic pattern.
    )
    return has_contract_interaction


# ---------------------------------------------------------------------------
# Change analysis
# ---------------------------------------------------------------------------

def extract_changed_function_names(contract_before: str, contract_after: str) -> Set[str]:
    """Extract changed function names from before and after contract ASTs.

    Covers functions, constructors, fallback functions, and receive functions.
    """
    try:
        from .test_ast import _get_parser
    except ImportError:
        from reconstruct_dataset.test_ast import _get_parser

    if _get_parser is None:
        # Fall back to regular expressions.
        return _extract_changed_functions_regex(contract_before, contract_after)

    try:
        before_funcs = _ts_extract_function_signatures(contract_before or "")
        after_funcs = _ts_extract_function_signatures(contract_after or "")
    except Exception:
        return _extract_changed_functions_regex(contract_before, contract_after)

    # Added, removed, or signature-changed functions.
    added = after_funcs - before_funcs
    removed = before_funcs - after_funcs

    # For common signatures, compare function bodies.
    common = before_funcs & after_funcs
    changed = set()
    for sig in common:
        before_body = _ts_extract_function_body(contract_before or "", sig)
        after_body = _ts_extract_function_body(contract_after or "", sig)
        if before_body != after_body:
            changed.add(sig)

    # Return plain function names, including constructor, fallback, and receive.
    result = set()
    for sig in added | removed | changed:
        name = sig.split("(", 1)[0]
        result.add(name)
    return result


def _ts_extract_function_signatures(sol_code: str) -> Set[str]:
    """Extract Solidity function signatures, including constructor, fallback, and receive."""
    from tree_sitter import Language, Parser

    try:
        from .test_ast import _get_parser
    except ImportError:
        from reconstruct_dataset.test_ast import _get_parser

    parser = _get_parser("solidity")
    tree = parser.parse(bytes(sol_code, "utf8"))
    source = bytes(sol_code, "utf8")

    sigs: Set[str] = set()
    _collect_function_sigs(tree.root_node, source, sigs)
    return sigs


def _collect_function_sigs(node, source: bytes, sigs: Set[str]):
    """Collect function signatures from the AST."""
    if node.type == "function_definition":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            # Extract parameters to build the signature.
            params = _extract_params_text(node, source)
            sigs.add(f"{name}({params})")
        else:
            # An unnamed function may be a legacy function() fallback.
            sigs.add("fallback()")
    elif node.type == "constructor_definition":
        params = _extract_params_text(node, source)
        sigs.add(f"constructor({params})")
    elif node.type in ("fallback_function_definition", "receive_function_definition"):
        sigs.add(f"{node.type.split('_')[0]}()")

    for child in node.children:
        _collect_function_sigs(child, source, sigs)


def _extract_params_text(node, source: bytes) -> str:
    """Extract parameter-list text."""
    for child in node.children:
        if child.type == "parameter_list":
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip("()")
    return ""


def _ts_extract_function_body(sol_code: str, sig: str) -> str:
    """Extract function body text for a signature."""
    from tree_sitter import Language, Parser

    try:
        from .test_ast import _get_parser
    except ImportError:
        from reconstruct_dataset.test_ast import _get_parser

    parser = _get_parser("solidity")
    tree = parser.parse(bytes(sol_code, "utf8"))
    source = bytes(sol_code, "utf8")

    name = sig.split("(", 1)[0]
    result = [""]

    def _find(node):
        target_type = "function_definition"
        if name == "constructor":
            target_type = "constructor_definition"
        elif name in ("fallback", "receive"):
            target_type = f"{name}_function_definition"

        if node.type == target_type:
            if name == "constructor" or name in ("fallback", "receive"):
                result[0] = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                return
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                fn_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                if fn_name == name:
                    result[0] = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                    return
        for child in node.children:
            _find(child)

    _find(tree.root_node)
    return result[0]


def _extract_changed_functions_regex(contract_before: str, contract_after: str) -> Set[str]:
    """Regular-expression fallback when Tree-sitter is unavailable."""
    func_re = re.compile(r"function\s+(\w+)\s*\(")

    before_funcs = {m.group(1) for m in func_re.finditer(contract_before or "")}
    after_funcs = {m.group(1) for m in func_re.finditer(contract_after or "")}

    added = after_funcs - before_funcs
    removed = before_funcs - after_funcs
    common = before_funcs & after_funcs

    changed = set()
    for name in common:
        before_body = _extract_function_body(contract_before or "", name)
        after_body = _extract_function_body(contract_after or "", name)
        if before_body != after_body:
            changed.add(name)

    return added | removed | changed


def _extract_function_body(code: str, func_name: str) -> str:
    """Extract a complete function body using a simplified matcher."""
    pattern = re.compile(
        rf"function\s+{re.escape(func_name)}\s*\([^)]*\)\s*(?:public|private|internal|external)?\s*(?:view|pure|payable)?\s*(?:returns\s*\([^)]*\))?\s*\{{"
    )
    match = pattern.search(code)
    if not match:
        return ""

    # Find the matching brace.
    start = match.end() - 1  # Points to the opening brace.
    depth = 0
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[start:i + 1]
    return ""


def test_references_contract(test_content: str, contract_name: str) -> bool:
    """Return whether a test explicitly references the specified contract."""
    if not test_content or not contract_name:
        return False
    # artifacts.require("ContractName")
    if re.search(rf'artifacts\.require\(\s*[\'"]({re.escape(contract_name)})[\'"]\s*\)', test_content):
        return True
    # import ... from "...ContractName..."
    if re.search(rf'import\s+.*?from\s+[\'"][^\'"]*{re.escape(contract_name)}[^\'"]*[\'"]', test_content):
        return True
    # ContractName.new() / .deployed() / .at()
    if re.search(rf'\b{re.escape(contract_name)}\s*\.\s*(?:new|deployed|at)\s*\(', test_content):
        return True
    return False


def test_diff_calls_changed_functions(test_diff: str, changed_functions: Set[str]) -> bool:
    """Return whether a test diff calls a changed function."""
    if not test_diff or not changed_functions:
        return False
    for line in test_diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            for fn in changed_functions:
                if re.search(rf'\b{re.escape(fn)}\b', line):
                    return True
    return False


def is_trivial_change(diff_text: str) -> bool:
    """Return whether a diff contains only formatting, whitespace, or comments."""
    if not diff_text:
        return True
    substantive_lines = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            # Remove the diff marker and surrounding whitespace.
            content = line[1:].strip()
            # Ignore empty and comment-only lines.
            if not content or content.startswith("//") or content.startswith("/*") or content.startswith("*"):
                continue
            substantive_lines += 1
    return substantive_lines == 0


def _contract_name_from_path(sol_path: Path) -> str:
    """Infer a contract name from the file stem."""
    return sol_path.stem


def detect_test_framework(repo: Path) -> str:
    """Detect the project's test framework."""
    repo_files = {p.name for p in repo.iterdir() if p.is_file()}

    if "foundry.toml" in repo_files:
        return "foundry"
    if "hardhat.config.js" in repo_files or "hardhat.config.ts" in repo_files:
        return "hardhat"
    if "truffle.js" in repo_files or "truffle-config.js" in repo_files:
        return "truffle"

    pkg_path = repo / "package.json"
    if pkg_path.exists():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
            all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if any("hardhat" in k for k in all_deps):
                return "hardhat"
            if any("truffle" in k for k in all_deps):
                return "truffle"
            if any("foundry" in k for k in all_deps):
                return "foundry"
        except Exception:
            pass

    return "unknown"


# ---------------------------------------------------------------------------
# File-level co-evolution pairing
# ---------------------------------------------------------------------------

CoevolutionRecord = Tuple[CommitChange, Path, CommitChange, Path, str]
# (sol_commit, sol_path, test_commit, test_path, commit_type)


def coevolution_pairs_filelevel(
    commits: List[CommitChange],
    sol_to_tests: Dict[Path, List[Path]],
    log_file=None,
) -> List[CoevolutionRecord]:
    """
    Build file-level co-evolution pairs.

    Rule 1 (same_commit): the contract and linked test change in one commit.
        The test must reference the contract.
    Rule 2 (delayed): a linked test changes within FILELEVEL_LOOKAHEAD commits.
        All three evidence conditions must hold.
    """
    pairs: List[CoevolutionRecord] = []
    seen: Set[Tuple[str, str]] = set()  # Deduplicate by Solidity and test paths.

    for i, c in enumerate(commits):
        for sol_path, test_paths in sol_to_tests.items():
            if sol_path not in c.changed_files:
                continue

            sol_change = c.changed_files[sol_path]

            # Skip trivial changes.
            if is_trivial_change(sol_change.diff):
                continue

            contract_name = _contract_name_from_path(sol_path)

            # ---- Rule 1: same commit. ----
            tests_in_same = [t for t in test_paths if t in c.changed_files]
            if tests_in_same:
                # Extract changed function names and verify relevance.
                changed_functions = extract_changed_function_names(
                    sol_change.before or "", sol_change.after or ""
                )

                for t in tests_in_same:
                    test_change = c.changed_files[t]
                    if is_trivial_change(test_change.diff):
                        continue

                    test_content = test_change.before or test_change.after or ""

                    # Verify that this is a real test file.
                    if not _is_actual_test_file(t, test_content):
                        continue

                    # Verify that the test references the contract.
                    if not test_references_contract(test_content, contract_name):
                        if DEBUG:
                            log_print(
                                f"  SKIP (same-commit): test {t} does not reference {contract_name}",
                                log_file,
                            )
                        continue

                    # Verify that the test diff calls a changed function.
                    if changed_functions:
                        if not test_diff_calls_changed_functions(
                            test_change.diff, changed_functions
                        ):
                            if DEBUG:
                                log_print(
                                    f"  SKIP (same-commit): test {t} diff does not call changed functions {changed_functions}",
                                    log_file,
                                )
                            continue

                    pair_key = (str(sol_path), str(t))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    pairs.append((c, sol_path, c, t, "same_commit"))
                continue  # Do not evaluate delayed updates after a same-commit match.

            # ---- Rule 2: delayed commit. ----
            changed_functions = extract_changed_function_names(
                sol_change.before or "", sol_change.after or ""
            )

            found = False
            for j in range(i + 1, min(i + 1 + FILELEVEL_LOOKAHEAD, len(commits))):
                cj = commits[j]

                # Stop if the same contract changes again.
                if sol_path in cj.changed_files:
                    break

                for t in test_paths:
                    if t not in cj.changed_files:
                        continue

                    test_change = cj.changed_files[t]
                    if is_trivial_change(test_change.diff):
                        continue

                    test_content = test_change.before or test_change.after or ""

                    # Verify that this is a real test file.
                    if not _is_actual_test_file(t, test_content):
                        continue

                    # Condition 1: the test references the changed contract.
                    if not test_references_contract(test_content, contract_name):
                        continue

                    # Condition 2: the test diff calls a changed function.
                    if changed_functions:
                        if not test_diff_calls_changed_functions(
                            test_change.diff, changed_functions
                        ):
                            continue

                    # Condition 3: no intervening change modifies the same contract.

                    pair_key = (str(sol_path), str(t))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    pairs.append((c, sol_path, cj, t, "delayed"))
                    found = True
                    break
                if found:
                    break

    return pairs


# ---------------------------------------------------------------------------
# Single-repository construction
# ---------------------------------------------------------------------------

def build_for_repo(repo: Path, output_file, log_file=None) -> int:
    # Repository metadata.
    repo_url = ""
    repo_owner = ""
    repo_name = repo.name
    if gitpython is not None:
        try:
            r = gitpython.Repo(repo)
            origin = r.remotes.origin.url if r.remotes and r.remotes.origin else ""
            repo_url, owner, repo_only = normalize_repo_url(origin)
            if owner and repo_only:
                repo_owner = owner
                repo_name = repo_only
        except Exception:
            pass
    repo_name_fmt = f"{repo_owner}__{repo_name}" if repo_owner else repo_name

    # Detect the test framework.
    test_framework = detect_test_framework(repo)

    # Collect files.
    files = collect_files(repo)
    sol_files = files["sol_files"]
    test_files = files["test_files"]
    if not sol_files or not test_files:
        return 0

    # Build the contract-to-test mapping.
    link_map = build_link_map(sol_files, test_files)
    if DEBUG:
        mapped = sum(1 for _k, v in link_map.items() if v)
        log_print(f"  sol_files={len(sol_files)} test_files={len(test_files)} mapped_sols={mapped}", log_file)

    # Convert paths to repository-relative paths.
    rel_sol = {p: p.relative_to(repo) for p in sol_files}
    rel_test = {p: p.relative_to(repo) for p in test_files}
    relevant = set(rel_sol.values()) | set(rel_test.values())

    # Collect commit changes.
    commits = collect_commit_changes(repo, relevant)
    if DEBUG:
        log_print(f"  commits_with_relevant_changes={len(commits)}", log_file)

    # Build the repository-relative sol_to_tests mapping.
    sol_to_tests: Dict[Path, List[Path]] = {}
    for sol_abs, tests_abs in link_map.items():
        if not tests_abs:
            continue
        sol_rel = rel_sol.get(sol_abs)
        if not sol_rel:
            continue
        tests_rel = [rel_test[t] for t in tests_abs if t in rel_test]
        if tests_rel:
            sol_to_tests[sol_rel] = tests_rel

    # File-level co-evolution pairing
    pairs = coevolution_pairs_filelevel(commits, sol_to_tests, log_file)
    if DEBUG:
        same = sum(1 for p in pairs if p[4] == "same_commit")
        delayed = sum(1 for p in pairs if p[4] == "delayed")
        log_print(f"  coevolution_pairs={len(pairs)} (same_commit={same}, delayed={delayed})", log_file)

    sample_count = 0
    for sol_commit, sol_path, test_commit, test_path, commit_type in pairs:
        sol_change = sol_commit.changed_files.get(sol_path)
        test_change = test_commit.changed_files.get(test_path)
        if not sol_change or not test_change:
            continue

        # The public runtime updates an existing test against the post-change
        # contract. Skip file deletions and newly added tests that cannot
        # satisfy that input contract. Contract additions remain valid because
        # the runtime explicitly supports an empty contract_file_before value.
        if (
            not sol_change.after
            or not sol_change.diff
            or not test_change.before
            or not test_change.after
            or not test_change.diff
        ):
            if DEBUG:
                log_print(
                    f"  SKIP incompatible runtime record: {sol_path.as_posix()} -> "
                    f"{test_path.as_posix()}",
                    log_file,
                )
            continue

        # Extract changed function names.
        changed_functions = extract_changed_function_names(
            sol_change.before or "", sol_change.after or ""
        )

        # Check whether the test diff calls a changed function.
        test_calls = test_diff_calls_changed_functions(
            test_change.diff, changed_functions
        )

        sample = {
            "repo_name": repo_name_fmt,
            "repo_url": repo_url,
            "test_framework": test_framework,
            "commit_type": commit_type,

            "production_file_path": f"{repo.name}/{sol_path.as_posix()}",
            "production_commit_SHA": sol_commit.sha,
            "production_commit_time": sol_commit.commit_time,

            "contract_file_before": sol_change.before or "",
            "contract_file_after": sol_change.after or "",
            "contract_diff": sol_change.diff,

            "test_file_path": f"{repo.name}/{test_path.as_posix()}",
            "test_commit_SHA": test_commit.sha,
            "test_commit_time": test_commit.commit_time,

            "test_file_before": test_change.before or "",
            "test_file_after": test_change.after or "",
            "test_diff": test_change.diff,

            "changed_functions": sorted(changed_functions),
            "test_calls_changed_functions": test_calls,
        }

        try:
            output_file.write(json.dumps(sample, ensure_ascii=False))
            output_file.write("\n")
            output_file.flush()
            sample_count += 1
            if DEBUG:
                log_print(
                    f"    #{sample_count} {commit_type} | {sol_path.as_posix()} -> {test_path.as_posix()} | "
                    f"fns={sorted(changed_functions) if changed_functions else '(none)'}",
                    log_file,
                )
        except Exception as e:
            log_print(f"  ERROR writing sample: {e}", log_file)

    return sample_count


# ---------------------------------------------------------------------------
# Legacy module entry point
# ---------------------------------------------------------------------------

def _select_repos(targets):
    if not targets:
        repos = [p for p in REPO_ROOT.iterdir() if p.is_dir()]
        return sorted(repos, key=lambda p: p.name.lower())
    repo_map = {p.name.lower(): p for p in REPO_ROOT.iterdir() if p.is_dir()}
    selected = []
    for target in targets:
        target_lower = target.lower()
        if target_lower in repo_map:
            selected.append(repo_map[target_lower])
    return selected


def main(target_repos=None):
    json_name, log_name = generate_output_filename(OUTPUT_DIR)
    json_path = OUTPUT_DIR / json_name
    log_path = OUTPUT_DIR / log_name

    total_count = 0

    with json_path.open("w", encoding="utf-8") as json_file, \
         log_path.open("w", encoding="utf-8") as log_file:

        log_print(f"Output file: {json_path}", log_file)
        log_print(f"Log file: {log_path}", log_file)
        log_print("=" * 80, log_file)

        for repo in _select_repos(target_repos if target_repos is not None else TARGET_REPOS):
            if not project_eligible(repo):
                continue
            try:
                log_print(f"Processing {repo.name}...", log_file)
                count = build_for_repo(repo, json_file, log_file)
                total_count += count
                log_print(f"  Wrote {count} samples from {repo.name}", log_file)
            except Exception as e:
                log_print(f"ERROR: Failed to process {repo.name}: {e}", log_file)
                import traceback
                log_print(traceback.format_exc(), log_file)

        log_print("=" * 80, log_file)
        log_print(f"Total: Wrote {total_count} samples to {json_path}", log_file)


if __name__ == "__main__":
    main()
