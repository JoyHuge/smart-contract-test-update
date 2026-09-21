# Dataset format

The dataset uses UTF-8 JSON Lines: one JSON object per line.

Each record must include these fields:

| Field | Description |
| --- | --- |
| `repo_name` | Repository identifier derived from the Git origin. |
| `repo_url` | Normalized public repository URL. |
| `test_framework` | `hardhat`, `truffle`, `foundry`, or an empty string for detection. |
| `production_file_path` | Contract path relative to the project root. |
| `test_file_path` | Test path relative to the project root. |
| `contract_file_before` | Complete contract source before the change. |
| `contract_file_after` | Complete contract source after the change. |
| `contract_diff` | Unified diff for the contract change. |
| `test_file_before` | Complete test source before the change. |
| `test_file_after` | Complete developer-written ground-truth test used for evaluation. |
| `test_diff` | Unified diff for the test change. |

The loader also accepts optional metadata such as commit identifiers,
`changed_functions`, `commit_type`, and `test_calls_changed_functions`.

`build_dataset.py` creates records from selected rows in
`docs/test-projects.csv`.
