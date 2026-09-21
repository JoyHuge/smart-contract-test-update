"""Tree-sitter based Solidity interface extraction for SCT-Agent.

The paper's AST perception step does not need a token-heavy dump of every
statement node. It needs deterministic interface facts extracted by walking
the real Solidity syntax tree: signatures, types, inheritance, mutability, and
modifiers. This module performs that traversal without regular expressions.
"""

from __future__ import annotations

import ctypes
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any


class SctSolidityAstParser:
    """Extract compact Solidity interface facts from a Tree-sitter AST."""

    PARSER_NAME = "tree-sitter-solidity"
    GRAMMAR_PACKAGE_VERSION = "1.2.11"
    SCHEMA_VERSION = 2
    CACHE_TAG = "tree_sitter_solidity_v2"
    _DECLARATION_KINDS = {
        "contract_declaration": "contract",
        "interface_declaration": "interface",
        "library_declaration": "library",
    }
    _CALLABLE_KINDS = {
        "function_definition": "function",
        "constructor_definition": "constructor",
        "fallback_receive_definition": "fallback_or_receive",
    }
    _PARAMETER_NODE_TYPES = {"parameter", "event_parameter", "error_parameter"}
    _MAX_REPORTED_PARSE_ERRORS = 50

    def __init__(self) -> None:
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_solidity
        except ImportError as exc:
            raise RuntimeError(
                "SCT-Agent AST modes require tree-sitter and "
                f"tree-sitter-solidity=={self.GRAMMAR_PACKAGE_VERSION}."
            ) from exc

        try:
            language_value = tree_sitter_solidity.language()
            if isinstance(language_value, Language):
                language = language_value
            else:
                try:
                    language = Language(language_value)
                except TypeError:
                    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
                    get_pointer.restype = ctypes.c_void_p
                    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                    pointer = get_pointer(
                        language_value, b"tree_sitter.Language"
                    )
                    if not pointer:
                        raise RuntimeError(
                            "Unable to load the Tree-sitter Solidity language"
                        )
                    language = Language(pointer)
            try:
                parser = Parser(language)
            except TypeError:
                parser = Parser()
                parser.language = language
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize the Tree-sitter Solidity grammar. "
                "Use compatible tree-sitter and tree-sitter-solidity packages."
            ) from exc

        self._parser = parser
        try:
            self._grammar_package_version = version("tree-sitter-solidity")
        except PackageNotFoundError:
            self._grammar_package_version = "unknown"

    def parse_source(self, source: str, label: str = "contract") -> dict[str, Any]:
        source_text = source or ""
        source_bytes = source_text.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        contracts = [
            self._parse_contract(node, source_bytes)
            for node in root.named_children
            if node.type in self._DECLARATION_KINDS
        ]
        parse_errors = self._collect_parse_errors(root, source_bytes)

        return {
            "schema_version": self.SCHEMA_VERSION,
            "parser": self.PARSER_NAME,
            "grammar_package_version": self._grammar_package_version,
            "label": label,
            "source_bytes": len(source_bytes),
            "has_parse_errors": bool(root.has_error),
            "parse_error_count": len(parse_errors),
            "parse_errors": parse_errors[: self._MAX_REPORTED_PARSE_ERRORS],
            "contract_count": len(contracts),
            "contracts": contracts,
        }

    def parse_to_json(self, source: str, label: str = "contract") -> str:
        return json.dumps(self.parse_source(source, label=label), indent=2, ensure_ascii=False)

    def _parse_contract(self, node, source: bytes) -> dict[str, Any]:
        body = node.child_by_field_name("body")
        members = list(body.named_children) if body is not None else []
        kind = self._DECLARATION_KINDS[node.type]

        return {
            "kind": kind,
            "name": self._field_text(node, "name", source),
            "abstract": any(child.type == "abstract" for child in node.children),
            "inherits": [
                self._normalized_text(child, source)
                for child in node.named_children
                if child.type == "inheritance_specifier"
            ],
            "state_variables": [
                self._parse_state_variable(member, source)
                for member in members
                if member.type == "state_variable_declaration"
            ],
            "functions": [
                self._parse_callable(member, source)
                for member in members
                if member.type in self._CALLABLE_KINDS
            ],
            "modifier_definitions": [
                self._parse_modifier_definition(member, source)
                for member in members
                if member.type == "modifier_definition"
            ],
            "events": [
                self._parse_event(member, source)
                for member in members
                if member.type == "event_definition"
            ],
            "errors": [
                self._parse_error(member, source)
                for member in members
                if member.type == "error_declaration"
            ],
            "structs": [
                self._parse_struct(member, source)
                for member in members
                if member.type == "struct_declaration"
            ],
            "enums": [
                self._parse_enum(member, source)
                for member in members
                if member.type == "enum_declaration"
            ],
            "user_defined_types": [
                self._parse_user_defined_type(member, source)
                for member in members
                if member.type == "user_defined_type_definition"
            ],
        }

    def _parse_state_variable(self, node, source: bytes) -> dict[str, Any]:
        modifiers = [
            self._node_text(child, source)
            for child in node.children
            if child.type in {"constant", "immutable"}
        ]
        return {
            "name": self._field_text(node, "name", source),
            "type": self._field_text(node, "type", source),
            "visibility": self._field_text(node, "visibility", source),
            "modifiers": modifiers,
            "override": self._first_named_child_text(node, "override_specifier", source),
            "initializer": self._field_text(node, "value", source),
        }

    def _parse_callable(self, node, source: bytes) -> dict[str, Any]:
        declaration_kind = self._CALLABLE_KINDS[node.type]
        if node.type == "function_definition":
            name = self._field_text(node, "name", source) or "<fallback>"
        elif node.type == "constructor_definition":
            name = "constructor"
        else:
            keyword_types = {child.type for child in node.children}
            if "receive" in keyword_types:
                declaration_kind = "receive"
                name = "receive"
            else:
                declaration_kind = "fallback"
                name = "fallback"

        params = self._parse_direct_parameters(node, source)
        return_type = node.child_by_field_name("return_type")
        returns = self._parse_direct_parameters(return_type, source) if return_type else []
        visibility = self._first_named_child_text(node, "visibility", source)
        mutability = self._first_named_child_text(node, "state_mutability", source)
        if not mutability:
            mutability = next(
                (
                    self._node_text(child, source)
                    for child in node.children
                    if child.type in {"payable", "view", "pure"}
                ),
                "",
            )

        modifier_nodes = [
            child for child in node.named_children if child.type == "modifier_invocation"
        ]
        modifier_names = [self._modifier_name(child, source) for child in modifier_nodes]
        if not mutability and "constant" in modifier_names:
            mutability = "constant"
        if not mutability:
            mutability = "nonpayable"
        modifiers = [
            self._normalized_text(child, source)
            for child in modifier_nodes
            if self._modifier_name(child, source) not in {"constant", "returns"}
        ]

        return {
            "kind": declaration_kind,
            "name": name,
            "params": params,
            "visibility": visibility,
            "mutability": mutability,
            "virtual": any(child.type == "virtual" for child in node.named_children),
            "override": self._first_named_child_text(node, "override_specifier", source),
            "modifiers": modifiers,
            "returns": returns,
            "signature": self._signature(name, params),
        }

    def _parse_modifier_definition(self, node, source: bytes) -> dict[str, Any]:
        return {
            "name": self._field_text(node, "name", source),
            "params": self._parse_direct_parameters(node, source),
            "virtual": any(child.type == "virtual" for child in node.named_children),
            "override": self._first_named_child_text(node, "override_specifier", source),
        }

    def _parse_event(self, node, source: bytes) -> dict[str, Any]:
        return {
            "name": self._field_text(node, "name", source),
            "params": self._parse_direct_parameters(node, source),
            "anonymous": any(child.type == "anonymous" for child in node.children),
        }

    def _parse_error(self, node, source: bytes) -> dict[str, Any]:
        return {
            "name": self._field_text(node, "name", source),
            "params": self._parse_direct_parameters(node, source),
        }

    def _parse_struct(self, node, source: bytes) -> dict[str, Any]:
        body = node.child_by_field_name("body")
        fields = []
        if body is not None:
            for member in body.named_children:
                if member.type != "struct_member":
                    continue
                fields.append(
                    {
                        "name": self._field_text(member, "name", source),
                        "type": self._field_text(member, "type", source),
                    }
                )
        return {"name": self._field_text(node, "name", source), "fields": fields}

    def _parse_enum(self, node, source: bytes) -> dict[str, Any]:
        body = node.child_by_field_name("body")
        values = []
        if body is not None:
            values = [
                self._normalized_text(value, source)
                for value in body.named_children
                if value.type == "enum_value"
            ]
        return {"name": self._field_text(node, "name", source), "values": values}

    def _parse_user_defined_type(self, node, source: bytes) -> dict[str, str]:
        name_node = node.child_by_field_name("name")
        underlying = next(
            (
                self._normalized_text(child, source)
                for child in node.named_children
                if child != name_node
            ),
            "",
        )
        return {
            "name": self._field_text(node, "name", source),
            "underlying_type": underlying,
        }

    def _parse_direct_parameters(self, node, source: bytes) -> list[dict[str, Any]]:
        if node is None:
            return []
        params = []
        for child in node.named_children:
            if child.type not in self._PARAMETER_NODE_TYPES:
                continue
            params.append(
                {
                    "index": len(params),
                    "type": self._field_text(child, "type", source),
                    "name": self._field_text(child, "name", source),
                    "storage": self._field_text(child, "location", source),
                    "indexed": any(
                        grandchild.type == "indexed" for grandchild in child.children
                    ),
                }
            )
        return params

    def _collect_parse_errors(self, root, source: bytes) -> list[dict[str, Any]]:
        errors = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_error or node.is_missing:
                errors.append(
                    {
                        "type": node.type,
                        "missing": bool(node.is_missing),
                        "start": self._point(node.start_point),
                        "end": self._point(node.end_point),
                        "text": self._normalized_text(node, source)[:160],
                    }
                )
            stack.extend(reversed(node.children))
        return errors

    def _field_text(self, node, field_name: str, source: bytes) -> str:
        child = node.child_by_field_name(field_name)
        return self._normalized_text(child, source) if child is not None else ""

    def _first_named_child_text(self, node, child_type: str, source: bytes) -> str:
        return next(
            (
                self._normalized_text(child, source)
                for child in node.named_children
                if child.type == child_type
            ),
            "",
        )

    def _modifier_name(self, node, source: bytes) -> str:
        identifier = next(
            (child for child in node.named_children if child.type == "identifier"),
            None,
        )
        return self._node_text(identifier, source) if identifier is not None else ""

    @staticmethod
    def _node_text(node, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _normalized_text(self, node, source: bytes) -> str:
        return " ".join(self._node_text(node, source).split())

    @staticmethod
    def _point(point) -> dict[str, int]:
        try:
            row, column = point.row, point.column
        except AttributeError:
            row, column = point
        return {"line": int(row) + 1, "column": int(column) + 1}

    @staticmethod
    def _signature(name: str, params: list[dict[str, Any]]) -> str:
        types = ",".join("".join(str(param.get("type", "")).split()) for param in params)
        return f"{name}({types})"
