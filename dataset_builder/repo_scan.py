from pathlib import Path
from typing import Dict, List, Tuple

from .config import CONTRACT_DIR_NAMES, TEST_DIR_NAMES, TEST_EXTS, SOL_EXT
from .utils import iter_files, read_text


def find_dirs(repo: Path) -> Tuple[List[Path], List[Path]]:
    contract_dirs = []
    test_dirs = []
    for p in repo.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name in CONTRACT_DIR_NAMES:
            contract_dirs.append(p)
        if name in TEST_DIR_NAMES:
            test_dirs.append(p)
    return contract_dirs, test_dirs


def collect_files(repo: Path) -> Dict[str, List[Path]]:
    contract_dirs, test_dirs = find_dirs(repo)
    sol_files = []
    test_files = []

    for d in contract_dirs:
        sol_files.extend(list(iter_files(d, {SOL_EXT})))

    for d in test_dirs:
        test_files.extend(list(iter_files(d, TEST_EXTS)))

    return {
        "contract_dirs": contract_dirs,
        "test_dirs": test_dirs,
        "sol_files": sol_files,
        "test_files": test_files,
    }


def project_eligible(repo: Path) -> bool:
    files = collect_files(repo)
    if not files["sol_files"]:
        return False
    if not files["test_files"]:
        return False

    # ensure at least one test file looks like code
    for tf in files["test_files"]:
        try:
            text = read_text(tf)
        except Exception:
            continue
        if len(text.strip()) > 0:
            return True
    return False


