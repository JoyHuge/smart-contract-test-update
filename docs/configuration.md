# Configuration

`prepare.py` validates one dataset record and its project image. `run.py` runs
the formal experiment.

## Environment file

Copy `.env.example` to `.env`. Configure the image and project path:

```dotenv
SCT_DOCKER_IMAGE=my-project-image:latest
SCT_PROJECT_DIR=/workspace/project
```

CLI options `--image` and `--project-dir` remain available as advanced
overrides. The CLI value has priority over the process environment and `.env`.

## Provider variables

Each selected model requires `<PREFIX>_API_KEY`, `<PREFIX>_BASE_URL`, and
`<PREFIX>_MODEL`. Supported prefixes are `DEEPSEEK`, `GLM`, `GPT`, `CLAUDE`,
`GEMINI`, and `QWEN`.

All endpoints must be OpenAI-compatible because the shared client uses
`ChatOpenAI`. All six provider aliases use the same generation settings.

## Preparation command

```bash
python prepare.py \
  --dataset datasets/example.jsonl \
  --record-index 0
```

Optional preparation overrides are `--image`, `--project-dir`, `--framework`,
`--output-dir`, and `--env-file`.

## Formal experiment command

```bash
python run.py \
  --dataset datasets/example.jsonl \
  --record-index 0 \
  --method SCT-Agent \
  --model gpt
```

Required experiment choices are the dataset, zero-based record index, method,
and model provider alias. `--method` and `--model` accept comma-separated
values. The normal workflow requires a successful matching `prepare.py` report.

Advanced options include:

- `--force-rerun`: ignore matching completed evaluations;
- `--aggregate-only`: update Excel from existing evaluations only;
- `--output-dir`: override `results/<dataset-name>/`;
- `--excel-path`: override `<output-dir>/test_results.xlsx`;
- `--framework`: provide a fallback when the dataset framework is unknown;
- `--skip-preparation-check`: bypass the readiness report requirement.

## Runtime settings

The `.env.example` file lists request timeouts, retries, the output-token
limit, and the test timeout.
