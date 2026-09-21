import re
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class ContractFile:
    """Information collected for one contract file."""
    path: str
    content: str
    is_interface: bool
    is_library: bool
    is_target: bool

class ContractProjectContext:
    """Collect and format context from a smart-contract project."""

    def __init__(self, project_root: str, contracts_dir: str = "contracts"):
        self.project_root = Path(project_root)
        self.contracts_dir = self.project_root / contracts_dir
        self.contracts_cache: Dict[str, ContractFile] = {}
        self.import_graph: Dict[str, Set[str]] = {}

    def get_all_contracts(self) -> List[str]:
        """Return relative paths for all Solidity contract files."""
        contracts = []
        if not self.contracts_dir.exists():
            return contracts

        for file_path in self.contracts_dir.rglob("*.sol"):
            relative = file_path.relative_to(self.project_root)
            contracts.append(str(relative))

        return sorted(contracts)

    def extract_imports(self, content: str) -> List[str]:
        """Extract import paths from Solidity source."""
        imports = []
        patterns = [
            r'import\s+"([^"]+)"\s*;',
            r'import\s+[^;]+\s+from\s+"([^"]+)"\s*;',
            r'import\s+\{[^}]+\}\s+from\s+"([^"]+)"\s*;',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)

        return imports

    def analyze_file_type(self, path: str, content: str) -> Dict[str, bool]:
        """Classify a Solidity file by its contents."""
        lower_path = path.lower()
        is_interface = 'interface ' in content and (
            'i' in lower_path.split('/')[-1] or 'interface' in lower_path
        )
        is_library = 'library ' in content or 'lib' in lower_path.split('/')[-1]

        return {
            'is_interface': is_interface,
            'is_library': is_library
        }

    def load_contract(self, relative_path: str) -> Optional[ContractFile]:
        """Load one contract file."""
        full_path = self.project_root / relative_path
        if not full_path.exists():
            return None

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Error reading {relative_path}: {e}")
            return None

        types = self.analyze_file_type(relative_path, content)

        contract = ContractFile(
            path=relative_path,
            content=content,
            is_interface=types['is_interface'],
            is_library=types['is_library'],
            is_target=False
        )

        self.contracts_cache[relative_path] = contract
        return contract

    def resolve_import_path(self, import_path: str, current_file: str) -> Optional[str]:
        """Resolve an import to a project-relative path."""
        if import_path.startswith('@'):
            return None

        current_dir = Path(current_file).parent

        candidates = [
            str(current_dir / import_path),
            str(self.contracts_dir / import_path),
            import_path
        ]

        for candidate in candidates:
            full_path = self.project_root / candidate
            if full_path.exists():
                return str(full_path.relative_to(self.project_root))

        return None

    def build_import_graph(self, target_path: str, max_depth: int = 5):
        """Build the contract import dependency graph."""
        visited = set()

        def traverse(path: str, depth: int):
            if depth > max_depth or path in visited:
                return

            visited.add(path)
            self.import_graph[path] = set()

            contract = self.load_contract(path)
            if not contract:
                return

            imports = self.extract_imports(contract.content)
            for import_path in imports:
                resolved = self.resolve_import_path(import_path, path)
                if resolved:
                    self.import_graph[path].add(resolved)
                    traverse(resolved, depth + 1)

        traverse(target_path, 0)

    def get_related_contracts(
        self,
        target_path: str,
        include_interfaces: bool = True,
        include_libraries: bool = True,
        max_files: int = 20
    ) -> List[ContractFile]:
        """Return all contracts related to the target contract."""
        self.build_import_graph(target_path)

        related = []
        visited = set()

        def collect(path: str, priority: int):
            if path in visited:
                return

            visited.add(path)
            contract = self.contracts_cache.get(path)
            if not contract:
                return

            if contract.is_interface and not include_interfaces:
                return
            if contract.is_library and not include_libraries:
                return

            related.append((priority, contract))

            for dep in self.import_graph.get(path, set()):
                collect(dep, priority + 1)

        collect(target_path, 0)
        related.sort(key=lambda x: x[0])
        result = [contract for _, contract in related[:max_files]]

        for contract in result:
            if contract.path == target_path:
                contract.is_target = True

        return result

    def get_project_structure_summary(self) -> str:
        """Return a summary of the project structure."""
        all_contracts = self.get_all_contracts()

        if not all_contracts:
            return "No contracts found in the project."

        structure = {}
        for contract_path in all_contracts:
            parts = Path(contract_path).parts
            if len(parts) >= 2:
                dir_path = '/'.join(parts[:-1])
                if dir_path not in structure:
                    structure[dir_path] = []
                structure[dir_path].append(parts[-1])

        summary = []
        summary.append(f"Total contracts: {len(all_contracts)}")
        summary.append("\nProject structure:")

        for dir_path, files in sorted(structure.items()):
            summary.append(f"  {dir_path}/ ({len(files)} files)")
            for file in sorted(files)[:3]:
                summary.append(f"    - {file}")
            if len(files) > 3:
                summary.append(f"    ... and {len(files) - 3} more")

        return '\n'.join(summary)

    def get_context_for_test(
        self,
        target_contract_path: str,
        max_context_length: int = 15000
    ) -> str:
        """Build contract context for test generation."""
        related = self.get_related_contracts(target_contract_path)

        if not related:
            return "No related contracts found."

        context_parts = []
        context_parts.append("=== SMART CONTRACT PROJECT CONTEXT ===")
        context_parts.append(f"\nTarget Contract: {target_contract_path}")
        context_parts.append(f"Related Contracts: {len(related)} files\n")

        total_length = 0

        for contract in related:
            if total_length + len(contract.content) > max_context_length:
                context_parts.append(f"\n... (context truncated, reached {max_context_length} chars limit)")
                break

            marker = "★ TARGET CONTRACT" if contract.is_target else "•"
            if contract.is_interface:
                marker += " [INTERFACE]"
            elif contract.is_library:
                marker += " [LIBRARY]"

            context_parts.append(f"\n{'='*60}")
            context_parts.append(f"{marker} {contract.path}")
            context_parts.append('='*60)
            context_parts.append(contract.content)

            total_length += len(contract.content)

        context_parts.append(f"\n{'='*60}")
        context_parts.append(f"Total context length: {total_length} characters")

        return '\n'.join(context_parts)

    def get_function_signatures(self, contract_path: str) -> List[Dict[str, str]]:
        """Extract function signatures from a contract."""
        contract = self.load_contract(contract_path)
        if not contract:
            return []

        signatures = []
        pattern = r'(?:function\s+(\w+)\s*\((.*?)\)\s*(external|public|internal|private)(?:\s*(view|pure))?)'
        matches = re.finditer(pattern, contract.content, re.DOTALL)

        for match in matches:
            signatures.append({
                'name': match.group(1),
                'params': match.group(2).strip(),
                'visibility': match.group(3),
                'mutability': match.group(4) or 'nonpayable'
            })

        return signatures

    def get_all_contract_signatures(self, target_path: str) -> Dict[str, List[Dict]]:
        """Return signatures for all related contracts."""
        related = self.get_related_contracts(target_path)
        signatures = {}

        for contract in related:
            sigs = self.get_function_signatures(contract.path)
            if sigs:
                signatures[contract.path] = sigs

        return signatures

    def get_contracts_by_names(
        self,
        contract_names: List[str],
        max_context_length: int = 15000,
    ) -> str:
        """
        Build source context for a list of contract names.

        Args:
            contract_names: Contract names referenced by the test file.
            max_context_length: Maximum context length in characters.

        Imports and references are followed to include indirect dependencies
        without exceeding the requested context limit.
        """
        
        name_to_path = {}
        for sol_path in self.get_all_contracts():
            content = self.load_contract(sol_path)
            if not content:
                continue
            
            for match in re.finditer(
                r'\b(?:contract|library|interface)\s+(\w+)', content.content
            ):
                name_to_path[match.group(1)] = sol_path

        
        collected = {}  # path -> content
        for name in contract_names:
            path = name_to_path.get(name)
            if path and path not in collected:
                contract = self.load_contract(path)
                if contract:
                    collected[path] = contract.content

        
        indirect_names = set()
        for path, content in list(collected.items()):
            
            for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]*)\b', content):
                candidate = match.group(1)
                if (candidate in name_to_path
                        and name_to_path[candidate] not in collected
                        and candidate not in contract_names):
                    indirect_names.add(candidate)

        for name in indirect_names:
            path = name_to_path.get(name)
            if path and path not in collected:
                contract = self.load_contract(path)
                if contract:
                    collected[path] = contract.content

        if not collected:
            return "No matching contracts found."

        
        all_names = sorted(name_to_path.keys())
        parts = []
        parts.append("=== ALL CONTRACTS IN THIS PROJECT ===")
        parts.append(', '.join(all_names))
        parts.append("")

        total = 0
        for path, content in sorted(collected.items()):
            if total + len(content) > max_context_length:
                parts.append(f"\n... (truncated, reached {max_context_length} chars)")
                break
            parts.append(f"\n{'=' * 60}")
            parts.append(f"Contract: {path}")
            parts.append('=' * 60)
            parts.append(content)
            total += len(content)

        
        if indirect_names:
            parts.append(f"\n{'=' * 60}")
            parts.append("ADDITIONAL CONTRACTS (referenced but not fully loaded):")
            for name in sorted(indirect_names):
                path = name_to_path.get(name)
                if path and path not in collected:
                    contract = self.load_contract(path)
                    if contract:
                        
                        sigs = re.findall(
                            r'function\s+(\w+)\s*\(([^)]*)\)',
                            contract.content,
                        )
                        parts.append(f"\n  {name} ({path}):")
                        for fname, params in sigs:
                            parts.append(f"    function {fname}({params})")

        return '\n'.join(parts)
