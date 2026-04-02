# ASR Fine-Tuning, Modules, and RL Adaptation (Survey Paper)

Research code and documentation for a **survey / empirical paper** on **domain adaptation for speech-to-text (ASR / STT)**:

1. **Supervised fine-tuning (SFT)** on domain audio (healthcare, formal parliamentary speech).
2. **Second-stage fine-tuning with reward signals** derived from WER, domain-weighted WER (WWER), or an **LLM-based scorer (Google Gemini)** — implemented as an **auxiliary loss** on top of CTC (see Methods in your paper for precise terminology vs classical MWER in ASR literature).

## Claims / goals

- Compare **reward designs** (MWER, WWER, LLM) after a strong **SFT baseline**.
- Run a **fair framework comparison** (this repo’s NeMo path vs a separate HuggingFace / Whisper pipeline in notebooks or future scripts).
- Report **WER**, **entity-focused EWER**, **CER/SER**, **domain-term F1**, training cost, and **catastrophic forgetting** on LibriSpeech-style general speech.

## Repository layout

| Path | Purpose |
|------|---------|
| [`docs/project_summary_and_gcp_plan.md`](docs/project_summary_and_gcp_plan.md) | Full paper checklist, datasets, metrics, GCP plan, known pitfalls |
| [`docs/GCP_DEPLOYMENT_GUIDE.md`](docs/GCP_DEPLOYMENT_GUIDE.md) | Step-by-step GPU VM setup, **GCS bucket** `adaptive-ai-487419-stt-results`, cost notes |
| [`gcp_scripts/nemo_afrispeech_training.py`](gcp_scripts/nemo_afrispeech_training.py) | **NeMo** `EncDecCTCModelBPE`: SFT + reward stage, eval, JSON results |
| [`requirements-nemo-train.txt`](requirements-nemo-train.txt) | Pip dependencies for NeMo training |
| [`smoke_tests/`](smoke_tests/) | Legacy / exploratory notebooks |

## Datasets (NeMo script)

- **AfriSpeech-200 clinical:** official HF **`train` / `validation` / `test`** splits, filtered to `domain == "clinical"` (streaming-friendly). Default = use **all** clinical clips per split (override with `--train_samples`, `--val_samples`, `--test_samples`).
- **VoxPopuli (EN):** **10,000** training utterances sampled with **`--seed`** (not sequential), official **validation** for dev metrics. Frame as *parliamentary / formal* speech in the paper, not courtroom audio.
- **LibriSpeech clean-100:** used for **catastrophic forgetting** evaluation (validation manifest), optional training cap via config.

## Quickstart (local smoke test)

NeMo 2.x needs **Python 3.10+** (`python3.11 -m venv .venv`). Always install with the **same** interpreter you use to run the script (`python -m pip` avoids “pip installed numpy but python can’t import it”):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-nemo-train.txt
python gcp_scripts/nemo_afrispeech_training.py --smoke_test
```

If `pip install` says numpy is installed but `import numpy` still fails, run `python -c "import sys; print(sys.executable)"` and confirm it ends with `.../.venv/bin/python`; otherwise use `./.venv/bin/python -m pip install ...`.

If you see **`Dataset scripts are no longer supported, but found afrispeech-200.py`**, your `datasets` package is 3.x. Reinstall with: `python -m pip install "datasets>=2.14.0,<3.0.0"` (already pinned in `requirements-nemo-train.txt`).

If transcribe / dataloader fails with **`np.sctypes` removed in NumPy 2.0**, NeMo’s audio code is not NumPy-2-ready yet: `python -m pip install "numpy>=1.22,<2.0"` (pinned in `requirements-nemo-train.txt`).

Artifacts: `checkpoints/`, `manifests/`, `results/*_results.json`.

## GCP training

Full **onboarding** (billing, APIs, GPU quota, bucket, `tmux` so your **laptop can sleep** while the **VM** trains): [`docs/GCP_DEPLOYMENT_GUIDE.md`](docs/GCP_DEPLOYMENT_GUIDE.md) — includes a **paper vs script** checklist.

On the VM, use **`tmux`** (or `nohup`), then:

```bash
export GEMINI_API_KEY=...   # only for --real_llm / --reward_mode llm
python gcp_scripts/nemo_afrispeech_training.py --stage both \
  --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/my_experiment \
  --seed 42
```

**RL from existing SFT checkpoint:**

```bash
python gcp_scripts/nemo_afrispeech_training.py --stage rl \
  --sft_checkpoint ./checkpoints/sft_model.nemo \
  --reward_mode wwer
```

**Useful flags:** `--use_lora` (best-effort NeMo encoder adapter), `--mock_llm` / `--real_llm`, `--skip_zero_shot`, `--skip_librispeech_forgetting`, `--skip_test_eval`, `--voxpopuli_train_subset N`.

## Results persistence

- **`results/<run_id>_results.json`**: config snapshot, zero-shot / SFT / RL metrics (WER, CER, SER, EWER, domain F1), reward trajectory, optional LibriSpeech forgetting, optional **paired bootstrap** p-value, test-split metrics.
- **`--upload_gcs`**: copies checkpoints and JSON to **Google Cloud Storage** so work survives VM deletion.

## Citation

When public, cite the AfriSpeech-200 and VoxPopuli dataset papers and your forthcoming manuscript. Internal project GCP ID: **adaptive-ai-487419**.
