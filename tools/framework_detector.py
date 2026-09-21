from pathlib import Path
from typing import Optional

class FrameworkDetector:
    """Detect the test framework used by a project."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def detect(self) -> Optional[str]:
        """Return the detected framework name, or ``None``."""
        
        files_to_check = [
            'hardhat.config.js',
            'hardhat.config.ts',
            'truffle.js',
            'truffle-config.js',
            'foundry.toml',
            'package.json'
        ]

        for file_name in files_to_check:
            file_path = self.project_root / file_name
            if file_path.exists():
                if file_name == 'foundry.toml':
                    return 'foundry'
                elif file_name == 'package.json':
                    return self._check_package_json()
                elif 'hardhat' in file_name:
                    return 'hardhat'
                elif 'truffle' in file_name:
                    return 'truffle'

        
        return None

    def _check_package_json(self) -> Optional[str]:
        """Infer the framework from package.json dependencies."""
        package_json_path = self.project_root / 'package.json'
        if not package_json_path.exists():
            return None

        try:
            import json
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)

            dependencies = package_data.get('dependencies', {})
            dev_dependencies = package_data.get('devDependencies', {})

            
            all_deps = {**dependencies, **dev_dependencies}

            
            if '@nomicfoundation/hardhat-toolbox' in all_deps or 'hardhat' in all_deps:
                return 'hardhat'
            elif 'truffle' in all_deps:
                return 'truffle'
            elif 'foundry' in all_deps:
                return 'foundry'

        except Exception as e:
            print(f"⚠️  Error reading package.json: {e}")

        return None
