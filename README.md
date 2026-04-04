# ASR Fine-Tuning, Modules, and RL Adaptation (Survey Paper)

Research code and documentation for a **survey / empirical paper** on **domain adaptation for speech-to-text (ASR / STT)** across multiple toolkits.

## Goals

1. **Supervised fine-tuning (SFT)** on domain audio — healthcare (AfriSpeech-200 clinical) and formal parliamentary speech (VoxPopuli).
2. **Reward-augmented second-stage fine-tuning** using reward signals derived from WER, domain-weighted WER (WWER), or an **LLM-based scorer (Google Gemini)** — implemented as an auxiliary loss on top of CTC.
3. **Fair cross-framework comparison** between NVIDIA NeMo and ESPNet, using identical datasets, evaluation metrics, and reward designs.

## Research questions

- Do reward designs (MWER, WWER, LLM) consistently improve over a strong SFT baseline across frameworks?
- Which framework offers a better trade-off between domain-adaptation quality and training cost?
- How much catastrophic forgetting occurs on general-domain speech (LibriSpeech) after domain fine-tuning?

## Metrics

**WER**, entity-focused **EWER**, **CER / SER**, **domain-term F1**, training cost (GPU-hours), and catastrophic forgetting delta on LibriSpeech clean-100.

## Repository layout

```
.
├── data/       # Shared datasets and manifests (AfriSpeech-200, VoxPopuli, LibriSpeech)
├── nemo/       # NVIDIA NeMo experiments — SFT + reward-augmented training
└── espnet/     # ESPNet experiments (forthcoming)
```

| Path | Purpose |
|------|---------|
| [`data/`](data/) | Single source of truth for all audio manifests and dataset preparation notes |
| [`nemo/`](nemo/) | NeMo `EncDecCTCModelBPE` training scripts, notebooks, GCP deployment guides |
| [`espnet/`](espnet/) | ESPNet recipes and training scripts (planned) |

## Progress

| Component | Status |
|-----------|--------|
| NeMo SFT baseline (AfriSpeech clinical) | In progress |
| NeMo SFT baseline (VoxPopuli) | In progress |
| NeMo reward-augmented training (MWER / WWER / LLM) | In progress |
| NeMo catastrophic-forgetting eval (LibriSpeech) | In progress |
| ESPNet SFT baseline | Planned |
| ESPNet reward-augmented training | Planned |
| Cross-framework comparison & paper write-up | Planned |

## Datasets

All datasets are shared between frameworks. See [`data/README.md`](data/README.md) for download and preparation instructions.

| Dataset | Domain | Role |
|---------|--------|------|
| AfriSpeech-200 (clinical) | Healthcare | Primary adaptation target |
| VoxPopuli (EN) | Parliamentary | Secondary adaptation target |
| LibriSpeech clean-100 | General | Catastrophic-forgetting evaluation |

## Getting started

NeMo 2.x needs **Python 3.10+** (`python3.11 -m venv .venv`). Always install with the **same** interpreter you use to run the script (`python -m pip` avoids "pip installed numpy but python can't import it"):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-nemo-train.txt
python nemo/gcp_scripts/nemo_afrispeech_training.py --smoke_test
```

If `pip install` says numpy is installed but `import numpy` still fails, run `python -c "import sys; print(sys.executable)"` and confirm it ends with `.../.venv/bin/python`; otherwise use `./.venv/bin/python -m pip install ...`.

If you see **`Dataset scripts are no longer supported, but found afrispeech-200.py`**, your `datasets` package is 3.x. Reinstall with: `python -m pip install "datasets>=2.14.0,<3.0.0"` (already pinned in `requirements-nemo-train.txt`).

If transcribe / dataloader fails with **`np.sctypes` removed in NumPy 2.0`**, NeMo's audio code is not NumPy-2-ready yet: `python -m pip install "numpy>=1.22,<2.0"` (pinned in `requirements-nemo-train.txt`).

Artifacts: `checkpoints/`, `manifests/`, `results/*_results.json`.

- **NeMo:** see [`nemo/README.md`](nemo/README.md) for environment setup, quickstart, and GCP training instructions.
- **ESPNet:** see [`espnet/README.md`](espnet/README.md) for planned work and setup (forthcoming).

## GCP training

Full **onboarding** (billing, APIs, GPU quota, bucket, `tmux` so your **laptop can sleep** while the **VM** trains): [`nemo/docs/GCP_DEPLOYMENT_GUIDE.md`](nemo/docs/GCP_DEPLOYMENT_GUIDE.md) — includes a **paper vs script** checklist.

On the VM, use **`tmux`** (or `nohup`), then:

```bash
export GEMINI_API_KEY=...   # only for --real_llm / --reward_mode llm
python nemo/gcp_scripts/nemo_afrispeech_training.py --stage both \
  --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/my_experiment \
  --seed 42
```

**RL from existing SFT checkpoint:**

```bash
python nemo/gcp_scripts/nemo_afrispeech_training.py --stage rl \
  --sft_checkpoint ./checkpoints/sft_model.nemo \
  --reward_mode wwer
```

**Useful flags:** `--use_lora` (best-effort NeMo encoder adapter), `--mock_llm` / `--real_llm`, `--skip_zero_shot`, `--skip_librispeech_forgetting`, `--skip_test_eval`, `--voxpopuli_train_subset N`.

## Results persistence

- **`results/<run_id>_results.json`**: config snapshot, zero-shot / SFT / RL metrics (WER, CER, SER, EWER, domain F1), reward trajectory, optional LibriSpeech forgetting, optional **paired bootstrap** p-value, test-split metrics.
- **`--upload_gcs`**: copies checkpoints and JSON to **Google Cloud Storage** so work survives VM deletion.

## Citation

When public, cite the AfriSpeech-200 and VoxPopuli dataset papers and the forthcoming manuscript. Internal project GCP ID: **adaptive-ai-487419**.
