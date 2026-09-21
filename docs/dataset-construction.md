# Dataset construction

`build_dataset.py` creates JSONL samples from smart-contract repositories.

## Select projects

`docs/test-projects.csv` contains project names and GitHub URLs. Project
indices are zero-based.

```bash
python build_dataset.py \
  --project-indices 0 3 7 \
  --output datasets/selected-projects.jsonl
```

The builder validates indices before cloning. Repeated indices are processed
once.

## What the builder does

For each selected repository, the builder:

1. Clones or reuses the repository.
2. Locates Solidity contracts and supported test files.
3. Links contract and test files.
4. Extracts relevant contract and test updates from Git history.
5. Writes complete before/after files and diffs to a JSONL dataset.

The default checkout location is `projects/<project_name>/`. Generated datasets
are written to `datasets/` unless `--output` is supplied.

## Output

Each generated record includes project metadata, contract and test paths,
before/after source files, diffs, commit identifiers, and changed-function
metadata.

The builder prints a summary for every generated record. Use the record index
and test commit when preparing a project image for an experiment.
