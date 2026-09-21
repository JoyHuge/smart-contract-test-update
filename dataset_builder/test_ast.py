import ctypes
from typing import Optional, Set

from .config import DEBUG

try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript
    import tree_sitter_solidity
    import tree_sitter_typescript
except Exception as exc:  # pragma: no cover - runtime environment
    Language = None
    Parser = None
    tree_sitter_javascript = None
    tree_sitter_solidity = None
    tree_sitter_typescript = None
    _TS_IMPORT_ERROR = exc

# Parser cache keyed by language name.
_parser_cache: dict = {}


def _ensure_tree_sitter():
    if (
        Language is None
        or Parser is None
        or tree_sitter_javascript is None
        or tree_sitter_solidity is None
        or tree_sitter_typescript is None
    ):
        raise RuntimeError(
            "Tree-sitter and the JavaScript, TypeScript, and Solidity language "
            "packages are required. Install the packages in requirements.txt."
        ) from _TS_IMPORT_ERROR


def _language_for(lang: str) -> Language:
    _ensure_tree_sitter()
    if lang == "javascript":
        capsule = tree_sitter_javascript.language()
    elif lang == "typescript":
        capsule = tree_sitter_typescript.language_typescript()
    elif lang == "solidity":
        capsule = tree_sitter_solidity.language()
    else:
        raise ValueError(f"Unsupported lang: {lang}")

    try:
        return Language(capsule)
    except TypeError:
        # tree-sitter 0.22 expects the pointer stored inside the language
        # package's PyCapsule. This branch keeps the parser compatible with
        # CodeBLEU 0.7, which requires tree-sitter below 0.23.
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        pointer = get_pointer(capsule, b"tree_sitter.Language")
        if not pointer:
            raise RuntimeError(f"Unable to load Tree-sitter language: {lang}")
        return Language(pointer)


def _get_parser(lang: str, log_file=None) -> Parser:
    """Return a cached parser for the requested language."""
    if lang in _parser_cache:
        return _parser_cache[lang]

    _ensure_tree_sitter()
    parser = Parser(_language_for(lang))
    _parser_cache[lang] = parser
    return parser


def _wrap_code(code: str, lang: str) -> str:
    if lang in {"javascript", "typescript"}:
        return "function __wrap__() {\n" + code + "\n}"
    if lang == "solidity":
        return "contract __T { function __wrap__() public {\n" + code + "\n} }"
    return code


def _collect_call_names(node, source_bytes: bytes, names: Set[str]):
    if node.type == "call_expression":
        func_node = node.child_by_field_name("function")
        if func_node is not None:
            name = _extract_callee_name(func_node, source_bytes)
            if name:
                names.add(name)

    for child in node.children:
        _collect_call_names(child, source_bytes, names)


def _extract_callee_name(node, source_bytes: bytes) -> Optional[str]:
    # member_expression: object.property
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        if prop is not None and prop.type == "property_identifier":
            return source_bytes[prop.start_byte:prop.end_byte].decode("utf-8", errors="ignore")
    # identifier: direct call fn(...)
    if node.type == "identifier":
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
    return None


def extract_called_function_names(code: str, lang: str, log_file=None) -> Set[str]:
    parser = _get_parser(lang, log_file)
    wrapped = _wrap_code(code, lang)
    tree = parser.parse(bytes(wrapped, "utf8"))
    names: Set[str] = set()
    _collect_call_names(tree.root_node, bytes(wrapped, "utf8"), names)
    return names


def hunk_calls_function(test_src: str, test_tgt: str, fn_name: str, ext: str, log_file=None) -> bool:
    ext = ext.lower()
    if ext == ".js":
        # Use TypeScript parser for JavaScript files due to version compatibility
        lang = "typescript"
    elif ext == ".ts":
        lang = "typescript"
    elif ext == ".sol":
        lang = "solidity"
    else:
        return False

    code = test_src + "\n" + test_tgt
    try:
        if DEBUG:
            msg = f"[tree-sitter] parse {lang} hunk for fn={fn_name}"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()
        names = extract_called_function_names(code, lang, log_file)
    except Exception:
        return False
    return fn_name in names
