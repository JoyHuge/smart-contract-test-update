import re
from pathlib import Path
from typing import Dict, List, Set, Optional

from .config import TEST_NAME_PREFIXES, TEST_NAME_SUFFIXES, CONTRACT_DIR_NAMES
from .utils import read_text, strip_comments_sol

try:
    from .test_ast import _get_parser
except Exception:
    _get_parser = None


# ---------------------------------------------------------------------------
# Tree-sitter AST extraction helpers
# ---------------------------------------------------------------------------

def _ts_extract_contract_names(sol_code: str) -> Set[str]:
    """Extract contract, interface, and library names from Solidity with Tree-sitter."""
    if _get_parser is None:
        return set()
    try:
        parser = _get_parser("solidity")
        tree = parser.parse(bytes(sol_code, "utf8"))
    except Exception:
        return set()

    names: Set[str] = set()
    _collect_contract_declarations(tree.root_node, bytes(sol_code, "utf8"), names)
    return names


def _collect_contract_declarations(node, source_bytes: bytes, names: Set[str]):
    """Collect names from contract, interface, and library declaration nodes."""
    if node.type in ("contract_declaration", "interface_declaration", "library_declaration"):
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            names.add(source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore"))
    for child in node.children:
        _collect_contract_declarations(child, source_bytes, names)


def _ts_extract_import_paths(test_code: str, lang: str) -> Set[str]:
    """Extract import and require paths from JavaScript or TypeScript tests."""
    if _get_parser is None:
        return set()
    try:
        parser = _get_parser(lang)
        tree = parser.parse(bytes(test_code, "utf8"))
    except Exception:
        return set()

    paths: Set[str] = set()
    _collect_import_paths(tree.root_node, bytes(test_code, "utf8"), paths)
    return paths


def _collect_import_paths(node, source_bytes: bytes, paths: Set[str]):
    """Collect import paths and require() paths from the AST."""
    # ES6 import: read the string child of an import statement.
    if node.type == "import_statement":
        for child in node.children:
            if child.type == "string":
                val = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                val = val.strip("'\"")
                if val:
                    paths.add(val)

    # require() call: call_expression where function is identifier "require"
    #   OR member_expression like artifacts.require('X')
    if node.type == "call_expression":
        func_node = node.child_by_field_name("function")
        if func_node is not None:
            is_require = False
            if func_node.type == "identifier":
                func_name = source_bytes[func_node.start_byte:func_node.end_byte].decode("utf-8", errors="ignore")
                if func_name == "require":
                    is_require = True
            elif func_node.type == "member_expression":
                prop_node = func_node.child_by_field_name("property")
                if prop_node is not None:
                    prop_name = source_bytes[prop_node.start_byte:prop_node.end_byte].decode("utf-8", errors="ignore")
                    if prop_name == "require":
                        is_require = True

            if is_require:
                for child in node.children:
                    if child.type == "arguments":
                        for arg in child.children:
                            if arg.type == "string":
                                val = source_bytes[arg.start_byte:arg.end_byte].decode("utf-8", errors="ignore")
                                val = val.strip("'\"")
                                if val:
                                    paths.add(val)

    for child in node.children:
        _collect_import_paths(child, source_bytes, paths)


def _ts_extract_instantiated_contracts(test_code: str, lang: str) -> Set[str]:
    """Extract instantiated contract names from JavaScript or TypeScript tests.

    Supported patterns:
    - Contract.new(...)
    - Contract.deployed(...)
    - Contract.at(...)
    - new Contract(...)
    - ethers.getContractFactory('ContractName')
    - artifacts.require('ContractName')
    """
    if _get_parser is None:
        return set()
    try:
        parser = _get_parser(lang)
        tree = parser.parse(bytes(test_code, "utf8"))
    except Exception:
        return set()

    names: Set[str] = set()
    _collect_instantiations(tree.root_node, bytes(test_code, "utf8"), names)
    return names


def _collect_instantiations(node, source_bytes: bytes, names: Set[str]):
    """Collect contract instantiations from the AST."""
    # new Contract(...)
    if node.type == "new_expression":
        constructor_node = node.child_by_field_name("constructor")
        if constructor_node is not None and constructor_node.type == "identifier":
            name = source_bytes[constructor_node.start_byte:constructor_node.end_byte].decode("utf-8", errors="ignore")
            if name and name[0].isupper():
                names.add(name)

    # call_expression: Contract.new() / Contract.deployed() / Contract.at()
    #                  artifacts.require('X') / ethers.getContractFactory('X')
    if node.type == "call_expression":
        func_node = node.child_by_field_name("function")
        if func_node is not None:
            # member_expression: Contract.new / Contract.deployed / Contract.at
            if func_node.type == "member_expression":
                obj_node = func_node.child_by_field_name("object")
                prop_node = func_node.child_by_field_name("property")
                if obj_node is not None and prop_node is not None:
                    prop_name = source_bytes[prop_node.start_byte:prop_node.end_byte].decode("utf-8", errors="ignore")
                    obj_text = source_bytes[obj_node.start_byte:obj_node.end_byte].decode("utf-8", errors="ignore")

                    if prop_name in ("new", "deployed", "at"):
                        # Direct contract name: Contract.new().
                        if obj_node.type == "identifier" and obj_text[0].isupper():
                            names.add(obj_text)

                    # artifacts.require('X') / ethers.getContractFactory('X')
                    if prop_name in ("require", "getContractFactory"):
                        # Read string arguments.
                        for child in node.children:
                            if child.type == "arguments":
                                for arg in child.children:
                                    if arg.type == "string":
                                        val = source_bytes[arg.start_byte:arg.end_byte].decode("utf-8", errors="ignore")
                                        val = val.strip("'\"")
                                        if val and val[0].isupper():
                                            names.add(val)

    for child in node.children:
        _collect_instantiations(child, source_bytes, names)


def _ts_extract_used_contract_entities(test_code: str, lang: str) -> Set[str]:
    """Extract PascalCase entity names from entity.method() calls in JavaScript or TypeScript tests.

    This covers calls such as Contract.transfer() and Token.balanceOf().
    """
    if _get_parser is None:
        return set()
    try:
        parser = _get_parser(lang)
        tree = parser.parse(bytes(test_code, "utf8"))
    except Exception:
        return set()

    names: Set[str] = set()
    _collect_entity_calls(tree.root_node, bytes(test_code, "utf8"), names)
    return names


def _collect_entity_calls(node, source_bytes: bytes, names: Set[str]):
    """Collect method calls on PascalCase entities from the AST."""
    if node.type == "call_expression":
        func_node = node.child_by_field_name("function")
        if func_node is not None and func_node.type == "member_expression":
            obj_node = func_node.child_by_field_name("object")
            if obj_node is not None and obj_node.type == "identifier":
                obj_name = source_bytes[obj_node.start_byte:obj_node.end_byte].decode("utf-8", errors="ignore")
                if obj_name and obj_name[0].isupper():
                    names.add(obj_name)
    for child in node.children:
        _collect_entity_calls(child, source_bytes, names)


# ---------------------------------------------------------------------------
# Path matching (unchanged)
# ---------------------------------------------------------------------------

def _basename_matches(sol_path: Path, test_path: Path) -> bool:
    sol_base = sol_path.stem.lower()
    test_base = test_path.stem.lower()
    if test_base == sol_base:
        return True
    for p in TEST_NAME_PREFIXES:
        if test_base == f"{p}{sol_base}":
            return True
    for s in TEST_NAME_SUFFIXES:
        if test_base == f"{sol_base}{s}":
            return True
    if sol_base in test_base:
        return True
    return False


def path_matching(sol_files: List[Path], test_files: List[Path]) -> Dict[Path, Set[Path]]:
    mapping: Dict[Path, Set[Path]] = {s: set() for s in sol_files}
    test_index = {}
    for tf in test_files:
        test_index.setdefault(tf.parent, []).append(tf)

    for sol in sol_files:
        sol_rel = list(sol.parts)
        idxs = [i for i, p in enumerate(sol_rel) if p.lower() in CONTRACT_DIR_NAMES]
        if not idxs:
            continue
        idx = idxs[-1]
        subpath = Path(*sol_rel[idx + 1:]).with_suffix("")

        for test_dir, tfs in test_index.items():
            try:
                rel_parts = list(test_dir.parts)
                tidx = [i for i, p in enumerate(rel_parts) if p.lower() in {"test", "tests"}]
                if not tidx:
                    continue
                t_idx = tidx[-1]
                test_subdir = Path(*rel_parts[t_idx + 1:])
            except Exception:
                continue

            if test_subdir == subpath.parent:
                for tf in tfs:
                    if _basename_matches(sol, tf):
                        mapping[sol].add(tf)
    return mapping


# ---------------------------------------------------------------------------
# Name matching (Tree-sitter based, three-gate AND)
# ---------------------------------------------------------------------------

def _ext_to_lang(ext: str) -> str:
    """Map a file extension to a Tree-sitter language."""
    if ext in (".ts",):
        return "typescript"
    return "javascript"


def name_matching(sol_files: List[Path], test_files: List[Path]) -> Dict[Path, Set[Path]]:
    """Apply the three Tree-sitter gates: import, instantiation, and use."""
    mapping: Dict[Path, Set[Path]] = {s: set() for s in sol_files}

    # Precompute contract names for each Solidity file.
    sol_contracts: Dict[Path, Set[str]] = {}
    for sol in sol_files:
        try:
            sol_text = strip_comments_sol(read_text(sol))
        except Exception:
            continue
        sol_contracts[sol] = _ts_extract_contract_names(sol_text)

    for tf in test_files:
        try:
            test_text = read_text(tf)
        except Exception:
            continue

        ext = tf.suffix.lower()
        lang = _ext_to_lang(ext)

        # Extract all three forms of Tree-sitter evidence.
        import_paths = _ts_extract_import_paths(test_text, lang)
        instantiated = _ts_extract_instantiated_contracts(test_text, lang)
        used_entities = _ts_extract_used_contract_entities(test_text, lang)

        for sol in sol_files:
            contracts = sol_contracts.get(sol, set())
            if not contracts:
                continue

            # Gate A: an import or require path points to the Solidity file.
            sol_name = sol.stem
            has_import = False
            for p in import_paths:
                # Match a path suffix or an artifacts.require contract name.
                if p.endswith(f"{sol_name}.sol") or p.endswith(sol_name) or sol_name in p:
                    has_import = True
                    break
            if not has_import:
                continue

            # Gate B: the test instantiates the contract.
            if not (contracts & instantiated):
                continue

            # Gate C: the test calls a method on the contract entity.
            if not (contracts & used_entities):
                continue

            mapping[sol].add(tf)

    return mapping


# ---------------------------------------------------------------------------
# build_link_map: serial strategy without a fallback
# ---------------------------------------------------------------------------

def build_link_map(sol_files: List[Path], test_files: List[Path]) -> Dict[Path, Set[Path]]:
    # Step 1: path-based matching
    mapping = path_matching(sol_files, test_files)

    # Step 2: for unmatched contracts, try stricter name-based matching
    unmatched = [sol for sol in sol_files if not mapping.get(sol)]
    if unmatched:
        name_map = name_matching(unmatched, test_files)
        for sol, tests in name_map.items():
            if tests:
                mapping[sol] = tests

    return mapping
