# NeMo — ASR Domain Adaptation

This folder contains all NVIDIA NeMo experiments for the shared ASR domain-adaptation study. See the [root README](../README.md) for overall goals and cross-framework progress.

## Folder layout

| Path | Purpose |
|------|---------|
| [`gcp_scripts/nemo_afrispeech_training.py`](gcp_scripts/nemo_afrispeech_training.py) | `EncDecCTCModelBPE`: SFT + reward stage, eval, JSON results |
| [`requirements-nemo-train.txt`](requirements-nemo-train.txt) | Pip dependencies for NeMo training |
| [`docs/project_summary_and_gcp_plan.md`](docs/project_summary_and_gcp_plan.md) | Full paper checklist, datasets, metrics, GCP plan, known pitfalls |
| [`docs/GCP_DEPLOYMENT_GUIDE.md`](docs/GCP_DEPLOYMENT_GUIDE.md) | Step-by-step GPU VM setup, GCS bucket `adaptive-ai-487419-stt-results`, cost notes |
| [`smoke_tests/`](smoke_tests/) | Legacy and exploratory notebooks |
| [`NeMo_STT_Domain_Adaptation_SFT_+_RL_Training_Pipeline.ipynb`](NeMo_STT_Domain_Adaptation_SFT_+_RL_Training_Pipeline.ipynb) | Main SFT + RL training notebook |

## Datasets

Data is shared with the ESPNet experiments. Manifests and raw audio are kept under [`../data/`](../data/). See [`../data/README.md`](../data/README.md) for download and preparation instructions.

| Dataset | Role |
|---------|------|
| **AfriSpeech-200 (clinical)** | Primary domain-adaptation target — HF `train` / `validation` / `test`, filtered `domain == "clinical"` |
| **VoxPopuli (EN)** | Secondary domain target — 10 000 training utterances (parliamentary / formal speech) |
| **LibriSpeech clean-100** | Catastrophic-forgetting evaluation |

## Quickstart — local smoke test

Validates imports, data streaming, manifest build, NeMo or CPU training step, reward hook, and JSON output without spending GCP credits.

```bash
cd nemo/
pip install -r requirements-nemo-train.txt
python gcp_scripts/nemo_afrispeech_training.py --smoke_test
```

Artifacts land in `checkpoints/`, `manifests/`, and `results/*_results.json`.

## GCP training

1. Provision a GPU VM and GCS bucket (see [`docs/GCP_DEPLOYMENT_GUIDE.md`](docs/GCP_DEPLOYMENT_GUIDE.md)).
2. Run SFT + RL together:

```bash
export GEMINI_API_KEY=...   # required only for --real_llm / --reward_mode llm
python gcp_scripts/nemo_afrispeech_training.py --stage both \
  --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/my_experiment \
  --seed 42
```

3. RL from an existing SFT checkpoint:

```bash
python gcp_scripts/nemo_afrispeech_training.py --stage rl \
  --sft_checkpoint ./checkpoints/sft_model.nemo \
  --reward_mode wwer
```

**Useful flags:** `--use_lora` (encoder adapter), `--mock_llm` / `--real_llm`, `--skip_zero_shot`, `--skip_librispeech_forgetting`, `--skip_test_eval`, `--voxpopuli_train_subset N`.

## Results persistence

- **`results/<run_id>_results.json`** — config snapshot, zero-shot / SFT / RL metrics (WER, CER, SER, EWER, domain F1), reward trajectory, optional LibriSpeech forgetting, optional paired bootstrap p-value, test-split metrics.
- **`--upload_gcs`** — copies checkpoints and JSON to Google Cloud Storage so work survives VM deletion.

## Reward designs under test

| Mode flag | Reward signal |
|-----------|--------------|
| `--reward_mode mwer` | Minimum WER (MWER) — sentence-level expected WER |
| `--reward_mode wwer` | Domain-weighted WER (WWER) — upweights clinical / domain terms |
| `--reward_mode llm` | LLM-based scorer via Google Gemini API |
