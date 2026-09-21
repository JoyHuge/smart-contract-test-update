import json
import re
from pathlib import Path


_CONTRACT_ADDITION_HUNK_RE = re.compile(r"^@@ -0(?:,0)? \+\d+(?:,\d+)? @@", re.MULTILINE)


def is_contract_addition_record(record: dict | None) -> bool:
    """Return whether *record* represents a Solidity file added from empty Pold.

    An empty ``contract_file_before`` is valid only when the diff confirms a
    file addition and the new contract source is available.  This keeps
    corrupted records invalid while allowing the addition change type included
    in the experiment dataset.
    """
    record = record or {}
    before = str(record.get("contract_file_before") or "")
    after = str(record.get("contract_file_after") or "")
    diff = str(record.get("contract_diff") or "")
    return not before and bool(after) and bool(
        _CONTRACT_ADDITION_HUNK_RE.search(diff) or "new file mode" in diff
    )


class DatasetLoader:
    """Load contract and test changes from a file-level JSONL dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.records = []
        self.current_record = None
        self.total_records = 0

    def load_dataset(self) -> int:
        """Load all JSONL records, one JSON object per line."""
        self.records = []
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
            self.total_records = len(self.records)
            print(f"✅ Loaded {self.total_records} records from dataset")
            return self.total_records
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return 0

    def set_current_record(self, index: int) -> bool:
        """Select the record at the given zero-based index."""
        if 0 <= index < self.total_records:
            self.current_record = self.records[index]
            return True
        return False

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    def get_contract_file_before(self) -> str:
        """Return the complete contract file before the change."""
        if self.current_record:
            return self.current_record.get('contract_file_before', '')
        return ''

    def get_contract_file_after(self) -> str:
        """Return the complete contract file after the change."""
        if self.current_record:
            return self.current_record.get('contract_file_after', '')
        return ''

    def get_contract_diff(self) -> str:
        """Return the contract's Git diff."""
        if self.current_record:
            return self.current_record.get('contract_diff', '')
        return ''

    def get_changed_functions(self) -> list:
        """Return the names of changed functions."""
        if self.current_record:
            return self.current_record.get('changed_functions', [])
        return []

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    def get_test_file_before(self) -> str:
        """Return the complete test file before the change."""
        if self.current_record:
            return self.current_record.get('test_file_before', '')
        return ''

    def get_test_file_after(self) -> str:
        """Return the developer-written test used as evaluation ground truth."""
        if self.current_record:
            return self.current_record.get('test_file_after', '')
        return ''

    def get_test_diff(self) -> str:
        """Return the test file's Git diff."""
        if self.current_record:
            return self.current_record.get('test_diff', '')
        return ''

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    def get_production_before(self) -> str:
        """Compatibility alias for ``get_contract_file_before``."""
        return self.get_contract_file_before()

    def get_production_after(self) -> str:
        """Compatibility alias for ``get_contract_file_after``."""
        return self.get_contract_file_after()

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:
        """Return normalized metadata for the current record."""
        if not self.current_record:
            return {}

        prod_sha = self.current_record.get('production_commit_SHA', '')
        test_sha = self.current_record.get('test_commit_SHA', '')

        return {
            'repo_name': self.current_record.get('repo_name', ''),
            'test_framework': self.current_record.get('test_framework', ''),
            'production_file_path': self.current_record.get('production_file_path', ''),
            'test_file_path': self.current_record.get('test_file_path', ''),
            'production_commit_SHA': prod_sha,
            'test_commit_SHA': test_sha,
            'production_commit': prod_sha,
            'test_commit': test_sha,
            'commits_match': prod_sha == test_sha,
            'commit_type': self.current_record.get('commit_type', ''),
            'changed_functions': self.current_record.get('changed_functions', []),
            'test_calls_changed_functions': self.current_record.get('test_calls_changed_functions', False),
        }

    def validate_record(self) -> dict:
        """Validate the current record, including legitimate file additions."""
        if not self.current_record:
            return {'warnings': ['No record loaded']}

        warnings = []
        errors = []

        required_fields = [
            'contract_file_after',
            'contract_diff',
            'test_file_before',
            'test_file_after',
            'test_diff',
        ]

        for field in required_fields:
            if not self.current_record.get(field):
                errors.append(f'Missing required field: {field}')

        if not self.current_record.get('contract_file_before'):
            if is_contract_addition_record(self.current_record):
                warnings.append(
                    'Contract addition: contract_file_before is empty; '
                    'using empty Pold with contract_file_after and contract_diff'
                )
            else:
                errors.append('Missing required field: contract_file_before')

        if not self.get_metadata()['commits_match']:
            warnings.append(
                f"Different commits (contract={self.get_metadata()['commit_type']})"
            )

        if not self.current_record.get('test_calls_changed_functions', False):
            warnings.append("Test diff does not call changed functions")

        return {'warnings': warnings, 'errors': errors}
