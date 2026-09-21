"""Configuration constants used by the dataset builder."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "projects"
OUTPUT_DIR = PROJECT_ROOT / "datasets"

# The standalone public entry point passes repositories explicitly. These
# defaults retain compatibility with the original module entry point.
TARGET_REPOS = None

DEBUG = True
TEST_HUNK_TOP_K = 1

COMMIT_FILTER_KEYWORDS = ["clean up", "merge", "compiler warning"]

CONTRACT_DIR_NAMES = {"contract", "contracts"}
TEST_DIR_NAMES = {"test", "tests"}

TEST_EXTS = {".js", ".ts", ".sol"}
SOL_EXT = ".sol"

TEST_NAME_PREFIXES = ["test"]
TEST_NAME_SUFFIXES = ["test", "spec"]

MAX_LOOKAHEAD_COMMITS = 50
COEVOLUTION_FINAL_RATIO_MAX = 10
