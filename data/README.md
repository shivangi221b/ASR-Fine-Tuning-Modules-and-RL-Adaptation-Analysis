# Shared Data

This folder is the single source of truth for all datasets used across the NeMo and ESPNet experiments. Both framework sub-projects reference manifests and audio cached here.

## Datasets

### AfriSpeech-200 (Clinical)

- **Source:** HuggingFace dataset `tobiolatunji/afrispeech-200`
- **Splits used:** official `train`, `validation`, `test`
- **Filter:** `domain == "clinical"`
- **Role:** Primary domain-adaptation target (healthcare / clinical speech)
- **Notes:** Streaming-friendly via HF datasets. Override sample counts with `--train_samples`, `--val_samples`, `--test_samples` in the NeMo script.

### VoxPopuli (EN)

- **Source:** HuggingFace dataset `facebook/voxpopuli`, language `en`
- **Training subset:** 10 000 utterances sampled with a fixed `--seed` (not sequential)
- **Dev/eval:** official `validation` split
- **Role:** Secondary domain target — parliamentary / formal speech
- **Notes:** Do not frame as courtroom audio; frame as parliamentary speech in the paper.

### LibriSpeech clean-100

- **Source:** [openslr.org/12](https://www.openslr.org/12/) or HuggingFace `openslr/librispeech_asr`
- **Splits used:** `validation` (manifest only)
- **Role:** Catastrophic-forgetting evaluation on general-domain speech

## Directory structure (after preparation)

```
data/
├── manifests/          # NeMo-format JSON manifests (audio_filepath, duration, text)
│   ├── afrispeech_clinical_train.json
│   ├── afrispeech_clinical_val.json
│   ├── afrispeech_clinical_test.json
│   ├── voxpopuli_en_train.json
│   ├── voxpopuli_en_val.json
│   └── librispeech_clean100_val.json
└── audio/              # Raw audio (populated by dataset download scripts)
    ├── afrispeech/
    ├── voxpopuli/
    └── librispeech/
```

## Data preparation

Manifests are generated automatically when you run the NeMo training script for the first time:

```bash
cd nemo/
python gcp_scripts/nemo_afrispeech_training.py --smoke_test
```

For ESPNet (once available), corresponding preparation scripts will live in `../espnet/scripts/` and write manifests in the format expected by that toolkit.

## Data policy

Raw audio files are **not committed to git** (too large). Add `data/audio/` to `.gitignore`. Only JSON manifests and this README are tracked.
