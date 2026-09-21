# Docker image requirements

Prepare a Docker image for the selected dataset record and set its name in
`.env`.

A compatible image should provide:

- `/bin/bash` and Python 3
- Node.js and the Python packages in `requirements.txt`
- the target smart-contract project at `SCT_PROJECT_DIR`
- project dependencies and a working test command
- the project state and ground-truth test represented by the selected record

The runner mounts the agent code at `/opt/sct/agent`, the dataset at
`/data/dataset.jsonl`, and results at `/results`.

## Validation

Run `prepare.py` before a formal experiment. It checks the project path,
ground-truth test, framework, and test execution.

## Optional hooks

An image may provide these executable hooks:

- `/opt/sct/job-setup.sh`
- `/opt/sct/job-cleanup.sh`

Use hooks for project-specific setup, such as starting a local development
chain.
