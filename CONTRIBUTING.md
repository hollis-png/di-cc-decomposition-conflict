# Contributing

Thanks for your interest in this project. This repository is the code and
data release accompanying an academic paper on Decomposition-Induced
Context-Memory Conflict (DI-CC). It is a **research artifact**, not an
actively-developed product — please set your expectations accordingly:

- Issues and pull requests are welcome, but there is **no SLA** on response
  time. This repo is maintained on a best-effort basis alongside other work.
- The primary goal of this codebase is exact reproducibility of the paper's
  results, not general-purpose reusability. Changes that risk altering the
  frozen pipeline's numerical output (e.g. to `phase0/decomposition/`,
  `phase0/probe/`, or the prompts in `decompose.py`) will be scrutinized
  more heavily than changes to setup/tooling.

## What's a Good Contribution Here

- Bug fixes that don't change reproducible results (e.g. a crash on a valid
  input, a broken path on a platform we didn't test).
- Documentation fixes/clarifications (README, docstrings, this file).
- Packaging/tooling improvements (`requirements.txt`, `setup.sh`, CI).
- Reports of results you could not reproduce, with enough detail to
  investigate (see [Reporting Issues](#reporting-issues) below).

Changes to core methodology (prompts, labeling logic, probe training, layer
selection, statistical tests) are unlikely to be merged directly, since they
would diverge from what's described in the paper — but issues proposing or
discussing them are still welcome.

## Development Setup

```bash
git clone <this-repository-url>
cd di-cc-decomposition-conflict
./setup.sh
source .venv/bin/activate
```

See [README.md](README.md) for how to run the pipeline stages. Since most
stages need a CUDA GPU and download multi-GB models on first run, consider
using `--limit N` (supported by stages `01`, `04`, `24`, `25`) for a fast
smoke test before running a change against the full pipeline.

## Branch / PR Workflow

1. Fork the repository and create a branch off `main`.
2. Make your change. Keep commits focused and use clear commit messages.
3. If your change could affect any script's output, note in the PR
   description whether you re-ran the relevant stage(s) and what you
   observed (ideally with `--limit` for a quick check).
4. Open a pull request describing what changed and why.

## Code Style Notes

- The codebase targets **Python 3.10+** and uses PEP 604 union type hints
  (`int | None`) and built-in generics (`list[str]`) — please match this
  style rather than importing from `typing`.
- Pipeline stages live in `scripts/` as numbered, single-purpose entry
  points that read/write paths defined centrally in `config.py`. Shared
  logic belongs in `phase0/`, not duplicated across scripts.
- Long-running GPU scripts (`24`, `25`) follow an append + flush + skip-done
  pattern for resumability — please preserve this if you touch them.

## Reporting Issues

When filing a bug report, please include:

- The script/stage you ran and the exact command line.
- Whether you used the frozen data files (`data/factscore_subclaims_40person.jsonl`,
  `data/h3_bios_191person.jsonl`) or regenerated data yourself.
- Your environment: Python version, `torch`/`transformers`/`bitsandbytes`
  versions, GPU model, and OS.
- The full error traceback or, for a results-mismatch report, the relevant
  output from `outputs/reports/`.

See the issue templates under `.github/ISSUE_TEMPLATE/` for a structured
starting point.

## Using Claude Code

If you use [Claude Code](https://claude.com/claude-code) to explore or
modify this repository, it can read `config.py` and the numbered
`scripts/` files directly to understand the pipeline — there's no separate
agent-context file in this repo, so point it at `README.md` and `config.py`
first.
