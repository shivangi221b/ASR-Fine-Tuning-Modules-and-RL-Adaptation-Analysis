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

- **NeMo:** see [`nemo/README.md`](nemo/README.md) for environment setup, quickstart, and GCP training instructions.
- **ESPNet:** see [`espnet/README.md`](espnet/README.md) for planned work and setup (forthcoming).

## Citation

When public, cite the AfriSpeech-200 and VoxPopuli dataset papers and the forthcoming manuscript. Internal project GCP ID: **adaptive-ai-487419**.
