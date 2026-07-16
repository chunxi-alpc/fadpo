# Fa-DPO

Faithfulness-aware direct preference optimization for medical question answering.

This repository contains the reproducible research code and the latest manuscript sources for Fa-DPO. The method decomposes controlled negative responses into reasoning-function-labeled Atomic Reasoning Units (ARUs), derives verifier-based risk signals, and uses those signals for retained-negative selection, rejected-token weighting, and risk-adaptive preference margins.

## Repository layout

- `fa_dpo_pipeline/`: data construction, ARU decomposition, evidence routing, risk scoring, preference-data preparation, training, evaluation, and analysis code.
- `experiments/`: public experiment protocols and selected analysis scripts. Private annotations and raw audit data are intentionally excluded.
- `comparison/`: example manifests, model matrices, and templates for controlled evaluations.
- `paper/`: latest manuscript source, marked revision source, response letter, bibliography, LaTeX styles, and the figures required by the manuscript.
- `docs/`: public reproducibility notes and release boundaries.

## Quick start

Install the pipeline dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r fa_dpo_pipeline/requirements.txt
```

The pipeline expects input datasets and model services to be supplied by the user. They are not bundled in this repository. See `fa_dpo_pipeline/README.md` for the staged workflow and command-line interfaces.

## Paper sources

The current manuscript is `paper/main.tex`. The marked version is `paper/main_review_marked.tex`, and the detailed response letter is `paper/response_letter2.tex`. The paper directory includes the bibliography, LaTeX class/style files, and the referenced figure assets.

## Scope and data policy

This public repository contains code, templates, protocols, and manuscript sources only. It does not contain private clinical annotations, patient-containing records, raw or generated training corpora, model weights, checkpoints, runtime logs, API credentials, or server-specific paths. Verifier-derived process metrics are proxy evidence for research analysis and are not clinical ground truth or a deployment safety validation.

## Reproducibility

See `docs/REPRODUCIBILITY.md` for the expected runtime-directory conventions and release boundaries. Inputs and outputs should be kept outside version control.
