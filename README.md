# Smart Contract Test Update

Smart Contract Test Update is a framework for building smart-contract test
datasets and updating tests with large language models.

It supports SCT-Agent, its ablations, Static Direct Generation (SDG), and
Static Self-Refine (SSR).

## Requirements

- Python 3.10 or newer
- Git
- Docker with the Compose plugin
- A compatible Docker image for the selected project
- API settings for at least one model provider

Install dependencies and create a local configuration file:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

## Quick start

### 1. Build a dataset

Choose one or more project indices from
[`docs/test-projects.csv`](docs/test-projects.csv):

```bash
python build_dataset.py \
  --project-indices 0 \
  --output datasets/example.jsonl
```

### 2. Prepare a sample

Configure the project image and model settings in `.env`, then validate a
sample before running an experiment:

```bash
python prepare.py \
  --dataset datasets/example.jsonl \
  --record-index 0
```

### 3. Run an experiment

```bash
python run.py \
  --dataset datasets/example.jsonl \
  --record-index 0 \
  --method SCT-Agent \
  --model gpt
```

Results are written to `results/<dataset-name>/`, including an Excel workbook
named `test_results.xlsx`.

## Supported methods

- `SCT-Agent`
- `SCT-Agent-wo-TestRunner`
- `SCT-Agent-wo-AST-wo-TestRunner`
- `SDG`
- `SSR`

## Supported model aliases

- `deepseek`
- `glm`
- `gpt`
- `claude`
- `gemini`
- `qwen`

## Documentation

- [Configuration](docs/configuration.md)
- [Dataset construction](docs/dataset-construction.md)
- [Docker image requirements](docs/custom-image.md)
- [Dataset format](docs/dataset-format.md)
