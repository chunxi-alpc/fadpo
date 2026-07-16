# Reproducibility and release boundaries

The code is organized so that inputs and generated artifacts live outside the public source tree.

## Runtime convention

A runtime directory can contain:

```text
runtime-root/
  data/          # user-provided datasets and generated input corpora
  result/        # intermediate and scored records
  artifacts/     # indexes, metadata, and prepared preference data
  outputs/       # evaluation reports and plots
  checkpoints/   # local model adapters or merged checkpoints
  fa_dpo_pipeline/  # this repository's pipeline package
```

The scripts accept explicit input and output paths. When a script provides a runtime-root environment variable, set it to a private working directory rather than committing generated files.

## Services and models

Generation, judging, and training require user-supplied model paths or OpenAI-compatible services. Keep credentials in environment variables such as `OPENAI_API_KEY`; never place them in configuration files or command history shared with the repository.

## What is intentionally excluded

This release excludes raw and generated medical datasets, patient-containing examples, clinician annotation workbooks, private audit keys, model weights, checkpoints, large indexes, runtime logs, and server-specific deployment instructions. The manuscript reports aggregate results, while the repository provides the code and public-facing protocols needed to reproduce the analyses with authorized inputs.

Verifier-derived process metrics should be interpreted as proxy evidence for the study's constructs. They are not clinical ground truth and do not establish deployment safety.
