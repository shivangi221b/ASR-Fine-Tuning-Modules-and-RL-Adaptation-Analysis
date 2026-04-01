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

Validates imports, data streaming, manifest build, NeMo **or** CPU training step, reward hook, and JSON output **without** spending GCP credits:

```bash
pip install -r requirements-nemo-train.txt
python gcp_scripts/nemo_afrispeech_training.py --smoke_test
```

Artifacts: `checkpoints/`, `manifests/`, `results/*_results.json`.

## GCP training

1. Create a GPU VM and bucket (see [`docs/GCP_DEPLOYMENT_GUIDE.md`](docs/GCP_DEPLOYMENT_GUIDE.md)).
2. Run with durable uploads:

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
