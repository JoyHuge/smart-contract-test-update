import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple


def iter_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                yield p


def normalize_path(p: Path) -> str:
    return str(p.as_posix())


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def is_text_file(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
        return True
    except Exception:
        try:
            path.read_text(encoding="latin1")
            return True
        except Exception:
            return False


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin1", errors="ignore")


def strip_comments_sol(code: str) -> str:
    # simple removal for matching only
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return code


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def normalize_repo_url(url: str):
    url = url.strip()
    # ssh with protocol: ssh://git@github.com/owner/repo.git
    m = re.match(r"ssh://git@github\.com/(.+?)/(.+?)(?:\.git)?$", url)
    if m:
        owner, repo = m.group(1), m.group(2)
        return f"https://github.com/{owner}/{repo}", owner, repo
    # git@github.com:owner/repo.git
    m = re.match(r"git@github\.com:(.+?)/(.+?)(?:\.git)?$", url)
    if m:
        owner, repo = m.group(1), m.group(2)
        return f"https://github.com/{owner}/{repo}", owner, repo
    # https://github.com/owner/repo(.git)
    m = re.match(r"https?://github\.com/(.+?)/(.+?)(?:\.git)?$", url)
    if m:
        owner, repo = m.group(1), m.group(2)
        return f"https://github.com/{owner}/{repo}", owner, repo
    return url, "", ""


def normalize_code_lines(code: str) -> str:
    """
    Normalize code formatting to match total_data_filter.json style:
    - remove leading whitespace for each line
    - drop empty lines
    - keep internal whitespace and trailing whitespace as-is
    """
    if not code:
        return code
    lines = code.splitlines()
    normalized = [line.lstrip() for line in lines if line.strip() != ""]
    return "\n".join(normalized)
