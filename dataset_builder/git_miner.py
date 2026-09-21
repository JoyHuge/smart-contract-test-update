from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import COMMIT_FILTER_KEYWORDS, MAX_LOOKAHEAD_COMMITS

try:
    import git
except Exception as exc:  # pragma: no cover - runtime environment
    git = None
    _GIT_IMPORT_ERROR = exc


@dataclass
class FileChange:
    path: Path
    before: Optional[str]
    after: Optional[str]
    diff: str


@dataclass
class CommitChange:
    sha: str
    message: str
    commit_time: str
    changed_files: Dict[Path, FileChange]


def _filtered_message(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in COMMIT_FILTER_KEYWORDS)


def _ensure_gitpython():
    if git is None:
        raise RuntimeError(
            "GitPython is required to run this script in PyCharm. "
            "Please install it in your interpreter: pip install GitPython"
        ) from _GIT_IMPORT_ERROR


def list_commits(repo: Path) -> List[str]:
    _ensure_gitpython()
    r = git.Repo(repo)
    return [c.hexsha for c in r.iter_commits()]


def commit_message(repo: Path, sha: str) -> str:
    _ensure_gitpython()
    r = git.Repo(repo)
    return r.commit(sha).message.strip()


def commit_time(repo: Path, sha: str) -> str:
    _ensure_gitpython()
    r = git.Repo(repo)
    dt = r.commit(sha).committed_datetime
    # normalize to UTC ISO8601 with Z
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def changed_files(repo: Path, sha: str) -> List[Path]:
    _ensure_gitpython()
    r = git.Repo(repo)
    commit = r.commit(sha)
    paths = []
    for parent in commit.parents or []:
        diffs = commit.diff(parent)
        for d in diffs:
            if d.b_path:
                paths.append(Path(d.b_path))
            if d.a_path:
                paths.append(Path(d.a_path))
    # handle initial commit
    if not commit.parents:
        for item in commit.tree.traverse():
            if item.type == "blob":
                paths.append(Path(item.path))
    return list(dict.fromkeys(paths))


def file_content_at(repo: Path, sha: str, path: Path) -> Optional[str]:
    _ensure_gitpython()
    r = git.Repo(repo)
    try:
        blob = r.commit(sha).tree / path.as_posix()
        return blob.data_stream.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def file_diff(repo: Path, sha: str, path: Path) -> str:
    _ensure_gitpython()
    r = git.Repo(repo)
    commit = r.commit(sha)
    if commit.parents:
        parent = commit.parents[0]
        diff = parent.diff(commit, paths=path.as_posix(), create_patch=True)
    else:
        diff = commit.diff(git.NULL_TREE, paths=path.as_posix(), create_patch=True)
    if not diff:
        return ""
    return diff[0].diff.decode("utf-8", errors="ignore")


def collect_commit_changes(repo: Path, relevant_paths: Iterable[Path]) -> List[CommitChange]:
    # relevant_paths should be repo-relative Paths
    relevant_set = {Path(p) for p in relevant_paths}
    commits = list_commits(repo)
    results = []

    for sha in commits:
        msg = commit_message(repo, sha)
        if _filtered_message(msg):
            continue
        ctime = commit_time(repo, sha)
        files = changed_files(repo, sha)
        rel_files = [p for p in files if p in relevant_set]
        if not rel_files:
            continue

        changed: Dict[Path, FileChange] = {}
        for p in rel_files:
            before = file_content_at(repo, f"{sha}^", p)
            after = file_content_at(repo, sha, p)
            diff = file_diff(repo, sha, p)
            changed[p] = FileChange(path=p, before=before, after=after, diff=diff)
        results.append(CommitChange(sha=sha, message=msg, commit_time=ctime, changed_files=changed))
    return results


def _diff_line_count(diff_text: str) -> int:
    """Count changed diff lines, excluding file headers."""
    count = 0
    for line in diff_text.splitlines():
        if (line.startswith('+') and not line.startswith('+++')) or \
           (line.startswith('-') and not line.startswith('---')):
            count += 1
    return count


def _function_name_in_diff_lines(func_name: str, diff_text: str) -> bool:
    """Return whether a function name appears on a changed diff line."""
    for line in diff_text.splitlines():
        if line.startswith('+') or line.startswith('-'):
            if line.startswith('+++') or line.startswith('---'):
                continue
            if func_name in line:
                return True
    return False


COEVOLUTION_MAX_RATIO = 10  # Maximum test-to-production changed-line ratio.


def coevolution_pairs(
    commits: List[CommitChange],
    sol_to_tests: Dict[Path, List[Path]],
) -> List[Tuple[CommitChange, Path, CommitChange, Path]]:
    pairs = []

    # index commits by file change
    for i, c in enumerate(commits):
        for sol, tests in sol_to_tests.items():
            if sol not in c.changed_files:
                continue

            # if tests changed in same commit
            test_in_same = [t for t in tests if t in c.changed_files]
            if test_in_same:
                for t in test_in_same:
                    pairs.append((c, sol, c, t))
                continue

            # look ahead for delayed test update
            sol_change = c.changed_files[sol]
            sol_diff_lines = _diff_line_count(sol_change.diff or "")
            if sol_diff_lines == 0:
                continue

            found = False
            for j in range(i + 1, min(i + 1 + MAX_LOOKAHEAD_COMMITS, len(commits))):
                cj = commits[j]
                if sol in cj.changed_files:
                    # production changed again before test; no coevolution for current change
                    found = True
                    break
                for t in tests:
                    if t in cj.changed_files:
                        test_change = cj.changed_files[t]
                        test_diff_lines = _diff_line_count(test_change.diff or "")

                        # Condition A: changed-line ratio.
                        if test_diff_lines > sol_diff_lines * COEVOLUTION_MAX_RATIO:
                            continue

                        # Condition B: a changed test line must mention a production function.
                        # Collect production function names.
                        func_names = set()
                        import re
                        for match in re.finditer(r'function\s+(\w+)\s*\(', sol_change.after or ''):
                            func_names.add(match.group(1))
                        for match in re.finditer(r'function\s+(\w+)\s*\(', sol_change.before or ''):
                            func_names.add(match.group(1))

                        if func_names:
                            # At least one function name must occur on a changed test line.
                            if not any(_function_name_in_diff_lines(fn, test_change.diff or "") for fn in func_names):
                                continue

                        pairs.append((c, sol, cj, t))
                        found = True
                        break
                if found:
                    break
    return pairs
