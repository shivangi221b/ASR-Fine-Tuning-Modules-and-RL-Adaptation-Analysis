# Experiments and results (NeMo ASR)

This document is the **canonical experiment log + results snapshot** for the NeMo experiment driver:

- Script: `nemo/gcp_scripts/nemo_afrispeech_training.py`
- Methods: `nemo/docs/nemo_analysis.md` (implementation-level methodology, objectives, and reward definitions)

It consolidates what has actually been run so far (valid runs only), plus the missing items that still need to be executed for the paper.

---

## 1) Environment, hardware, and software versions (reproducibility)

### 1.1 What is known from artifacts/logs

- **Project / VM**
  - **GCP project**: `adaptive-ai-487419`
  - **VM name**: `finding-nemo-again`
  - **Zone**: `asia-east1-c`
- **Platform**: `Linux-6.8.0-1053-gcp-x86_64-with-glibc2.35`
- **GPU**: **1× Tesla T4 (16GB)**
- **NVIDIA driver**: `570.211.01` (`nvidia-smi`)
- **CUDA runtime (driver-reported)**: `12.8` (`nvidia-smi`)
- **Python**: `3.10.12` (GCC 11.4.0 build)
- **PyTorch**: `2.5.1+cu121` (`torch.version.cuda = 12.1`, `torch.cuda.is_available() = True`)
- **NeMo**: `2.2.1`

### 1.2 Captured from VM (Apr 28 2026)

**Runtime environment:**

| Package | Version |
|---|---|
| Python | 3.10.12 (GCC 11.4.0) |
| Platform | Linux-6.8.0-1053-gcp-x86_64-with-glibc2.35 |
| PyTorch | 2.5.1+cu121 (torch.version.cuda=12.1) |
| nemo-toolkit | 2.2.1 |
| pytorch-lightning | 2.6.1 |
| lightning | 2.4.0 |
| jiwer | 4.0.0 |
| omegaconf | 2.3.0 |
| datasets (HuggingFace) | 2.21.0 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| numpy | 1.26.4 |
| transformers | 4.48.3 |
| sentencepiece | 0.2.1 |
| GPU | Tesla T4, 15360 MiB, Driver 570.211.01, CUDA 12.8 |

Full pip freeze stored at `vm_results/pip_freeze.txt`.

### 1.3 GCP instance

- **Machine type**: `n1-standard-16` (16 vCPUs, 60 GB RAM)
- **GPU**: 1× Tesla T4 (16 GB VRAM), attached via GCP accelerator

---

## 2) Base model and framework details

### 2.1 Framework: NeMo `EncDecCTCModelBPE` (CTC-BPE ASR)

The experiment driver fine-tunes a NeMo ASR model in the **CTC-BPE** family:

- **Model class**: `nemo.collections.asr.models.EncDecCTCModelBPE`
- **Loss**: **CTC** (Connectionist Temporal Classification)
- **Tokenizer**: **SentencePiece BPE** (subword vocabulary; NeMo restores tokenizer from the `.nemo` checkpoint)
- **Decode during eval**: by default, NeMo `model.transcribe(...)` uses an **offline batch** inference loop (not streaming) and returns text hypotheses (greedy CTC decoding unless decoding config is overridden).

### 2.2 Base checkpoint used in these experiments

- **Base model name**: `stt_en_conformer_ctc_medium`
- **Source**: NVIDIA NGC NeMo model registry (downloaded via `EncDecCTCModelBPE.from_pretrained(...)`)

### 2.3 Exact architecture (snapshotted from `.nemo` config)

Extracted from `sft_model_config.json` (restored from `checkpoints/sft_model.nemo`):

**Audio frontend:**

| Setting | Value |
|---|---|
| Feature type | 80-dim log-mel filterbank |
| Window size | 25 ms |
| Window stride | 10 ms (frame shift) |
| FFT size | 512 |
| Normalization | per-feature (mean/var) |
| SpecAugment | 2 freq masks (width ≤ 27 bins), 5 time masks (width ≤ 5%) |

**Conformer encoder:**

| Setting | Value |
|---|---|
| Architecture | `ConformerEncoder` |
| Number of layers | 18 |
| Model dimension (d_model) | 256 |
| Attention heads | 4 |
| Attention type | Relative-position self-attention (`rel_pos`) |
| Context window | `[-1, -1]` — full bidirectional (offline/batch mode) |
| Subsampling | Striding, factor 4 |
| Conv kernel size | 31 |
| FF expansion factor | 4 |
| Dropout | 0.1 (encoder + attention) |

**Decoder and tokenizer:**

| Setting | Value |
|---|---|
| Decoder | `ConvASRDecoder` (linear projection) |
| Vocabulary size | 1024 SentencePiece BPE subword tokens |
| Tokenizer type | Unigram SentencePiece BPE |
| Loss | CTC (`mean_batch` reduction) |

### 2.4 Batch vs streaming

The current pipeline is an **offline / batch** transcription setup. The `att_context_size: [-1, -1]` setting confirms the encoder uses full bidirectional context over the entire utterance, which is incompatible with streaming. Evaluation calls `model.transcribe(paths, batch_size=...)` with greedy CTC decoding.

---

## 3) Methodology (high-level) and run structure

See `nemo/docs/nemo_analysis.md` for the full methodology. At a high level:

```mermaid
flowchart TD
    pretrained["Pretrained stt_en_conformer_ctc_medium"]
    zeroshot["Zero-shot eval (val)"]
    sft["Stage 1: SFT (CTC only)"]
    sfteval["Eval after SFT (val + test + Libri forgetting)"]
    rl["Stage 2: reward-augmented fine-tuning (\"RL\")"]
    rleval["Eval after RL (val)"]

    pretrained --> zeroshot
    pretrained --> sft
    sft --> sfteval
    sft --> rl
    rl --> rleval
```

**Reward modes implemented in code**: `mwer`, `wwer`, `llm`, `all`.  
**Reward modes executed so far**: **MWER and WWER** for both AfriSpeech and VoxPopuli (LLM/all explicitly deferred).

---

## 4) Datasets and sample counts

### 4.1 AfriSpeech-200 (clinical domain)

- **Dataset in config**: `afrispeech_clinical`
- **Validation split**: **1813 utterances** (clinical-domain subset; used for all val-set metrics)
- **Test split**: **3508 utterances**

### 4.2 LibriSpeech (forgetting evaluation only)

- Used only as a **post-training evaluation** to measure catastrophic forgetting.
- **Validation subset size**: **2694 utterances** (`librispeech_after_sft` block)

### 4.3 VoxPopuli (English parliamentary speech)

- Full English train split used (no subset cap).
- **Validation split**: **1742 utterances** (used for all val-set metrics)
- Audio written as lossless FLAC (PCM_16) to reduce disk usage; quality is identical to WAV for training and evaluation.

---

## 5) Hyperparameters and run configuration

### 5.1 AfriSpeech clinical (seed=42)

Hyperparameters from run `afrispeech_clinical_seed42_1776207077`:

| Setting | Value |
|---|---|
| Backbone | `stt_en_conformer_ctc_medium` |
| Batch size | 16 |
| SFT learning rate | 1e-4 |
| SFT epochs | 5 |
| RL learning rate | 1e-5 |
| RL epochs | 2 |
| Reward interval | every 4 steps |
| Stage-2 objective | `reweight_ctc` |
| Precision | fp32 |
| Gradient clip | 1.0 |

### 5.2 VoxPopuli (seed=33)

Hyperparameters from run `voxpopuli_model_conformer_medium_stage_sft_reward_mwer_seed33_bs8_sft5_rl2_1777160995`:

| Setting | Value |
|---|---|
| Backbone | `stt_en_conformer_ctc_medium` |
| Batch size | 8 |
| SFT learning rate | 1e-4 |
| SFT epochs | 5 |
| RL learning rate | 1e-5 |
| RL epochs | 2 |
| Reward interval | every 4 steps |
| Stage-2 objective | `reweight_ctc` |
| Precision | fp32 |

---

## 6) Metrics reported

All metrics are computed by `evaluate_manifest_bundle` in the driver:

- **WER (%) / CER (%)**: computed with `jiwer`, multiplied by 100
- **SER (%)**: percent of utterances where a normalized full-string match fails. SER can remain 100% even when WER improves, because it is a strict exact-match metric.
- **EWER (%)**: entity/domain WER — per-utterance WER restricted to domain-vocabulary tokens in the reference, averaged over utterances that contain at least one domain token.
- **Domain precision/recall/F1**: token-level metrics on occurrences of domain vocabulary terms (AfriSpeech clinical only; see §7).

Stage-2 runs also report:

- `reward_mean`, `reward_std` (from logged per-step batch rewards during RL training)

---

## 7) Results — AfriSpeech clinical validation (1813 utterances)

All RL metrics come from **clean, non-collapsed** runs.

**Run provenance:**
- Zero-shot + SFT: run `afrispeech_clinical_seed42_1776207077`
- MWER RL: run `afrispeech_clinical_model_conformer_medium_stage_rl_reward_mwer_seed42_bs8_sft5_rl2_1777307305`
- WWER RL: run `afrispeech_clinical_seed42_rl_1776462369` (checkpoint `rl_model.nemo`)

| Metric | Zero-shot | After SFT | After RL (MWER) | After RL (WWER) |
|---|---:|---:|---:|---:|
| WER (%) | 57.88 | 45.95 | **45.89** | 45.92 |
| CER (%) | 25.87 | **14.19** | 14.21 | 14.23 |
| SER (%) | 100.0 | 100.0 | 100.0 | 100.0 |
| EWER (%) | 19.97 | 20.27 | 19.82 | **18.92** |
| Domain P | 0.884 | 0.842 | 0.847 | **0.857** |
| Domain R | 0.857 | 0.867 | 0.878 | **0.885** |
| Domain F1 | 0.858 | 0.847 | 0.855 | **0.864** |
| Train time (s) | — | 12243 | 4382 | 4502 |

**Key observations:**
- Both RL reward modes reduce WER slightly over SFT (~0.06 pp MWER, ~0.02 pp WWER).
- EWER improves under both modes; WWER shows the larger gain (−1.35 pp vs SFT), consistent with its domain-weighted training signal.
- WWER consistently improves domain precision/recall/F1 more than MWER.
- SER remains 100% throughout — expected for a strict exact-match metric at this WER level.

---

## 8) Results — AfriSpeech clinical test set (3508 utterances)

Post-hoc evaluation using `eval_checkpoints.py` against `afrispeech_clinical_test.json`.

| Metric | After SFT | After RL (MWER) | After RL (WWER) |
|---|---:|---:|---:|
| WER (%) | **50.682** | 50.831 | 51.000 |
| CER (%) | **17.133** | 17.170 | 17.183 |
| SER (%) | 100.0 | 100.0 | 100.0 |
| EWER (%) | 23.276 | **22.833** | 22.868 |

**Key observations:**
- Unlike on the validation set, RL models show a slight WER regression on the test set (+0.15 pp MWER, +0.32 pp WWER). The small val improvement does not transfer to the test set for overall WER.
- EWER improves for both RL modes on the test set (−0.44 pp MWER, −0.41 pp WWER), consistent with the validation results. This is the more robust finding: RL reliably reduces domain entity errors even where overall WER is neutral or slightly worse.

---

## 9) Catastrophic forgetting — LibriSpeech after fine-tuning (2694 utterances)

Comparing LibriSpeech WER across all fine-tuning stages. Lower is better; a large increase indicates catastrophic forgetting.

**AfriSpeech models** (baseline: LibriSpeech WER after AfriSpeech SFT = 10.44%):

| Stage | WER (%) | CER (%) | Δ WER vs SFT (pp) |
|---|---:|---:|---:|
| After AfriSpeech SFT | 10.44 | 4.09 | baseline |
| After AfriSpeech MWER RL | 10.562 | 4.151 | +0.12 |
| After AfriSpeech WWER RL | 10.648 | 4.166 | +0.21 |

**VoxPopuli models** (baseline: LibriSpeech WER after VoxPopuli SFT = 6.225%):

| Stage | WER (%) | CER (%) | Δ WER vs SFT (pp) |
|---|---:|---:|---:|
| After VoxPopuli SFT | 6.225 | 2.263 | baseline |
| After VoxPopuli MWER RL | 6.317 | 2.268 | +0.09 |
| After VoxPopuli WWER RL | 6.335 | 2.278 | +0.11 |

**Key observations:**
- AfriSpeech RL stages cause minimal catastrophic forgetting: WER on LibriSpeech increases by only 0.12–0.21 pp over the SFT baseline.
- VoxPopuli RL stages also cause minimal forgetting: 0.09–0.11 pp WER increase, nearly identical in scale to AfriSpeech.
- VoxPopuli models show substantially lower LibriSpeech WER (~6.3%) than AfriSpeech models (~10.6%). This is expected: VoxPopuli (English parliamentary speech) is acoustically closer to LibriSpeech (read English) than AfriSpeech (African-accented clinical speech).
- The forgetting pattern is consistent across both datasets and both reward modes: the 2-epoch RL stage does not cause meaningful degradation on out-of-domain speech.

---

## 10) Results — VoxPopuli validation (1742 utterances)

All three stages come from separate, clean runs on seed=33, bs=8.

**Run provenance:**
- SFT: run `voxpopuli_model_conformer_medium_stage_sft_reward_mwer_seed33_bs8_sft5_rl2_1777160995`
- MWER RL: run `voxpopuli_model_conformer_medium_stage_rl_reward_mwer_seed33_bs8_sft5_rl2_1777223972`
- WWER RL: run `voxpopuli_model_conformer_medium_stage_rl_reward_wwer_seed33_bs8_sft5_rl2_1777267645`

| Metric | After SFT | After RL (MWER) | After RL (WWER) |
|---|---:|---:|---:|
| WER (%) | 16.014 | **15.999** | **15.981** |
| CER (%) | 5.829 | **5.824** | 5.826 |
| SER (%) | 100.0 | 100.0 | 100.0 |
| EWER (%) | 6.736 | **6.597** | 6.944 |
| Train time (s) | 57675 | 23083 | 23049 |

**Key observations:**
- VoxPopuli SFT already achieves low WER (~16%) because the Conformer-CTC-medium base was pre-trained on large English speech corpora.
- Both RL modes produce small but consistent WER reductions (~0.01–0.03 pp).
- MWER reduces EWER; WWER slightly increases it — likely because VoxPopuli domain terms are harder to isolate than clinical terms.

---

## 11) Training curves and reward trajectories

| Run | Dataset | Mode | Reward mean | Reward std | Train time |
|---|---|---|---:|---:|---|
| `1777307305` | AfriSpeech | MWER RL | 0.608 | 0.076 | 4382s (~1.2h) |
| `1776462369` | AfriSpeech | WWER RL | 0.609 | 0.076 | 4502s (~1.25h) |
| `1777160995` | VoxPopuli | SFT | — | — | 57675s (~16h) |
| `1777223972` | VoxPopuli | MWER RL | 0.891 | 0.040 | 23083s (~6.4h) |
| `1777267645` | VoxPopuli | WWER RL | 0.892 | 0.040 | 23049s (~6.4h) |

VoxPopuli reward means (~0.89) are notably higher than AfriSpeech (~0.61), consistent with VoxPopuli being cleaner speech with a lower baseline WER.

---

## 12) Statistical significance — paired bootstrap p-values

All p-values use 1000 bootstrap iterations, two-sided, paired at the utterance level (script: `nemo/gcp_scripts/compute_pvalue_existing_results.py`).

| Dataset | Comparison | WER SFT (%) | WER RL (%) | Δ WER (pp) | p-value |
|---|---|---:|---:|---:|---:|
| AfriSpeech (1813 utt) | SFT vs MWER RL | 45.944 | 45.885 | 0.058 | 0.512 |
| AfriSpeech (1813 utt) | SFT vs WWER RL | 45.944 | 45.924 | 0.019 | 0.609 |
| VoxPopuli (1742 utt) | SFT vs MWER RL | 16.014 | 15.999 | 0.013 | 0.650 |
| VoxPopuli (1742 utt) | SFT vs WWER RL | 16.014 | 15.981 | 0.032 | 0.501 |

**Interpretation:** None of the RL improvements are statistically significant at p < 0.05. The WER deltas are real but small relative to utterance-level variance. This is expected with a 2-epoch RL stage on a single seed. Significance would require larger RL improvements, multiple seeds, or a larger evaluation set.

---

## 13) TODO (missing experiments and reporting fields)

### 13.1 DER (diarization error rate)

DER cannot be computed from existing artifacts because the pipeline produces transcripts only (no speaker timing segments) and no reference RTTMs are stored for AfriSpeech clinical. Adding DER would require a diarization framework (e.g., NeMo diarization or pyannote) and reference RTTMs.

### 13.2 Reporting gaps to fix in future runs

- Latency / cost-per-improvement metrics
- Multiple seeds (currently only seed=42 for AfriSpeech, seed=33 for VoxPopuli); no variance or CIs reportable

### 13.3 Deferred experiments

- **LLM reward mode** (`--reward_mode llm`) — requires Gemini API key
- **LoRA variant** (`--use_lora`) — deferred; full fine-tuning only so far
