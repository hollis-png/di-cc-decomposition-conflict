---
name: Bug report
about: Report a crash, error, or results mismatch in the DI-CC pipeline
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear description of what went wrong.

## Stage / Script

Which numbered script (e.g. `scripts/05_label_spans.py`) or module were you
running?

## Command

```bash
# exact command line, including any flags (e.g. --limit)
```

## Steps to Reproduce

1.
2.
3.

## Expected Behavior

What did you expect to happen?

## Actual Behavior

What actually happened? Include the full error traceback if there is one:

```
paste traceback here
```

## Data Used

- [ ] Frozen data file (`data/factscore_subclaims_40person.jsonl` or `data/h3_bios_191person.jsonl`)
- [ ] Regenerated data (ran stage 04 / 24 myself)

If you regenerated data and are reporting a results mismatch, please
include the relevant output from `outputs/reports/` (e.g.
`transfer_eval.json`, `h3_mitigation.json`).

## Environment

- OS:
- Python version:
- `torch` version:
- `transformers` version:
- `bitsandbytes` version:
- GPU model / VRAM:
- CUDA version:

## Additional Context

Anything else that might help diagnose the issue.
