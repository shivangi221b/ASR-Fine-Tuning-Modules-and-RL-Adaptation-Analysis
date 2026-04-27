# ESPnet2 RL Extension: experiment driver + methods (paper-facing)

This document is the **paper-facing methods write-up** for this repository's ESPnet2 ASR reward-augmented fine-tuning experiment (the `egs2/afrispeech_rl/asr1/` recipe).

It covers the **two-stage methodology** (SFT → reward-augmented fine-tuning), the **reward definitions and objectives**, and the **hyperparameters** that define each run.

It intentionally **does not** include debugging narrative.

---

## 1. Purpose and high-level flow

**Recipe:** `egs2/afrispeech_rl/asr1/`  
**Orchestration script:** `run.sh`  
**Key model file:** `espnet2/asr/rl_espnet_model.py`  
**Key trainer file:** `espnet2/train/rl_trainer.py`

**Goal:** Domain adaptation for English **CTC + Attention** ASR using a **two-stage** procedure:

1. **Stage 1 — SFT:** Standard ESPnet2 training with **combined CTC + attention loss** (no reward term). The bottom 6 encoder blocks are frozen to preserve general acoustic representations.
2. **Stage 2 — Reward-augmented fine-tuning ("RL"):** Load the SFT checkpoint and fine-tune with a reward-driven objective. Rewards are computed from decoded hypotheses (non-differentiable) and incorporated into the stage-2 objective.

**Base model:** `pyf98/librispeech_conformer` (HuggingFace ESPnet Model Zoo; publicly accessible Conformer-BPE trained on LibriSpeech).

**Typical invocation (from `egs2/afrispeech_rl/asr1/`):**

```bash
# Full pipeline (stages 1–8)
bash run.sh --stage 1 --stop_stage 8 --ngpu 1

# SFT only (stages 6)
bash run.sh --stage 6 --stop_stage 6 --ngpu 1

# RL only (stage 7; requires SFT checkpoint)
bash run.sh --stage 7 --stop_stage 7 --ngpu 1

# Smoke test (small data, CPU-safe)
bash run.sh --stage 1 --stop_stage 8 --ngpu 0
```

**Run stages summary:**

| Stage | Purpose |
|-------|---------|
| 1 | Data preparation (AfriSpeech-200, VoxPopuli, LibriSpeech subset via HuggingFace streaming) |
| 2 | Feature dumping (copy raw audio to `dump/raw/`) |
| 3 | BPE token list check (populated by Stage 5) |
| 4 | Shape file generation (speech + text shapes for bucketed batching) |
| 5 | Pretrained model setup (`local/setup_pretrained.py`): download, extract token list / BPE / CMVN stats, patch YAML configs |
| 6 | SFT training (`espnet2_train.py --task asr_rl`) |
| 7 | RL training (`espnet2_train.py --task asr_rl`) |
| 8 | Decoding + extended evaluation (`local/eval_extended.py`) |

**Artifacts (per run):**

- SFT checkpoint: `exp/asr_sft/<run_name>/valid.acc.best.pth`
- RL checkpoint: `exp/asr_rl/<run_name>/valid.acc.best.pth`
- Training logs: `exp/asr_sft/train.log`, `exp/asr_rl/train.log`
- Decode outputs: `exp/asr_{sft,rl}/decode_<set>/text`
- Extended metrics: `exp/asr_{sft,rl}/decode_<set>/extended_metrics.json`
- Pretrained model info: `exp/pretrained/model_info.json`

---

## 1.1 What runs when (AfriSpeech vs LibriSpeech) and sample counts

**Training data** is a **combined** corpus of three sources:

- **AfriSpeech-200 (clinical domain)**: primary domain adaptation target
- **VoxPopuli (English)**: general English, prevents catastrophic forgetting of non-accented speech
- **LibriSpeech train.clean.100 (5k-utterance subset)**: clean read-speech anchor for forgetting evaluation

All three sources are combined into `data/train_combined` via sorted concatenation of Kaldi manifests.

**Smoke test sample counts:**

- Train: 30 utterances (AfriSpeech only, for Stage 1 data check)
- Val: 10 utterances
- Test: 5 utterances

**Full sample counts:**

- AfriSpeech train: all available clinical train utterances after 0.5–20 s duration filter (streamed from HuggingFace, not pre-downloaded)
- AfriSpeech dev: all available clinical validation utterances
- AfriSpeech test: **3,302 utterances** (used in reported results)
- VoxPopuli: up to 5,000 utterances (CLI-configurable `--max_samples`)
- LibriSpeech subset: 5,000 utterances (Stage 7 of `data.sh`)
- Combined train: ~15,000–18,000 utterances after filtering and combination
- LibriSpeech forgetting evaluation: **2,642 utterances** (`dev_clean` split, decoded post-SFT and post-RL)

---

## 2. Configuration and hyperparameters

### 2.1 SFT stage (`conf/train_asr_sft.yaml`)

| Area | Value | Notes |
|------|-------|-------|
| **Model** | `pyf98/librispeech_conformer` | Conformer encoder + Transformer decoder, CTC+Attn |
| **Encoder** | 12-block Conformer, `output_size=512`, `attention_heads=8`, `linear_units=2048` | Macaron-style, relative position encoding, CNN module |
| **Decoder** | 6-block Transformer, `attention_heads=8`, `linear_units=2048` | — |
| **Normalization** | `global_mvn` (CMVN stats from pretrained checkpoint) | Patched in by `local/setup_pretrained.py` |
| **Tokenizer** | SentencePiece BPE, 5000 vocab (`bpe_unigram5000`) | Extracted from pretrained checkpoint |
| **CTC weight** | 0.3 | Joint CTC+attention (`ctc_weight=0.3, lsm_weight=0.1`) |
| **Frozen layers** | Encoder blocks 0–5 (bottom 6 of 12) | `freeze_param` list in YAML |
| **SpecAugment** | Time warp ±5, 5× time mask (≤5% width), 2× freq mask (≤30 bins) | Applied during SFT |
| **Optimizer** | AdamW, `lr=1e-4`, `betas=(0.9,0.98)`, `weight_decay=1e-3` | — |
| **Scheduler** | WarmupLR, `warmup_steps=2000` | — |
| **Epochs** | 5 | — |
| **Batch type** | `numel`, `batch_bins=1,200,000` | Reduced from default to fit T4 16 GB with AMP |
| **Gradient accumulation** | 8 | Effective batch ≈ 9.6 M numel |
| **Gradient clipping** | 5.0 | — |
| **Precision** | AMP (fp16 mixed precision) | `use_amp: true`; halves memory vs fp32 |
| **Seed** | 42 | — |

### 2.2 RL stage (`conf/train_asr_rl.yaml`)

| Area | Value | Notes |
|------|-------|-------|
| **Init from** | Best SFT checkpoint | `init_param` set by `run.sh` Stage 7 |
| **Optimizer** | AdamW, `lr=1e-5`, same betas/weight\_decay | "1/10 SFT LR" heuristic (matches NeMo) |
| **Scheduler** | WarmupLR, `warmup_steps=500` | — |
| **Epochs** | 2 | — |
| **Batch type** | `numel`, `batch_bins=600,000` | Halved vs SFT; FP32 needs more headroom |
| **Gradient accumulation** | 8 | Effective batch ≈ 4.8 M numel |
| **Gradient clipping** | 1.0 | Tighter than SFT (reward stability; matches NeMo `GRAD_CLIP_VAL=1.0`) |
| **Precision** | FP32 | `use_amp: false`; matches NeMo `FORCE_FP32=True` |
| **Reward mode** | `wwer` | Weighted WER reward (domain-term emphasis) |
| **Reward weight** (`rl_weight`) | 0.02 | Scales reward contribution to loss |
| **Reward interval** | 4 | Compute reward every 4 batches; reuse cached otherwise |
| **Long-utterance guard** | `max_encoder_len_for_reward=1500` | Skip reward decode for very long sequences |
| **Stage-2 objective** | `reweight_ctc` | Gradient-affecting reweighted CTC (matches NeMo default) |
| **Domain term weight** | 3.0 | Clinical vocabulary tokens weighted 3× in WWER |
| **Domain terms** | 37 clinical tokens | `patient`, `hypertension`, `diabetes`, `medication`, … (see YAML) |
| **Seed** | 42 | — |

---

## 3. Data preparation and manifests

The recipe uses **Kaldi-style data directories** (`wav.scp`, `text`, `utt2spk`, `spk2utt`) rather than NeMo-style JSONL manifests. Raw audio is not pre-downloaded; instead, HuggingFace `datasets` streaming API is used to fetch utterances on demand.

**Data pipeline (`local/data.sh` + `local/data_hf.py`):**

1. Stream AfriSpeech-200 clinical split from HuggingFace (`tobiolatunji/afrispeech-200`, config `all`, `streaming=True`).
2. Stream VoxPopuli English from HuggingFace (`facebook/voxpopuli`, `streaming=True`).
3. Stream LibriSpeech `train.clean.100` (5k utterance subset) from HuggingFace (`openslr/librispeech_asr`, `streaming=True`).
4. Apply duration filtering (0.5 s – 20 s) in `data_hf.py`.
5. Sanitize utterance IDs: speaker-prefix all IDs (e.g., `{speaker}-{utt_id}`) to satisfy Kaldi's sort-by-speaker-ID requirement.
6. Write sorted Kaldi directories; validate each with `utils/validate_data_dir.sh --no-feats`.
7. Combine train sets via sorted concatenation + Python `spk2utt` rebuild (avoids `utils/combine_data.sh` fragility with mixed-source IDs).

**Feature dumping (Stage 2):** Audio paths are copied to `dump/raw/` directories; no filterbank extraction is done (the ESPnet2 model handles on-the-fly feature extraction using log-mel filterbanks from raw waveforms).

**Shape files (Stage 4):** Generated by an inline Python script in `run.sh` that reads `dump/raw/<set>/wav.scp` and the BPE-tokenized text to write `speech_shape` and `text_shape.bpe` files, bypassing `asr_train.py --collect_stats` (which does not recognize RL-specific config keys).

---

## 4. Stage 1 — Supervised fine-tuning (SFT)

**Entry:** `run.sh` Stage 6 → `espnet2_train.py --task asr_rl --config conf/train_asr_sft.yaml`

**Model loading:** `init_param` points to the pretrained `pyf98/librispeech_conformer` checkpoint (`.pth` file), downloaded and registered by `local/setup_pretrained.py`.

**Loss:** Combined CTC + cross-entropy attention loss:
$$L_\text{SFT} = (1 - \lambda_\text{ctc}) \cdot L_\text{att} + \lambda_\text{ctc} \cdot L_\text{ctc}$$
where $\lambda_\text{ctc} = 0.3$, `lsm_weight=0.1` (label smoothing).

**RL disabled for SFT:** `rl_weight: 0.0` in `train_asr_sft.yaml`; the reward path in `RLESPnetModel.forward()` is bypassed.

**Optimization:** AdamW + WarmupLR, fp16 AMP, gradient accumulation ×8, 5 epochs.

**After SFT:** Best checkpoint selected by `valid.acc` (attention decoder accuracy on validation set). Stage 8 runs greedy-CTC decoding on all test sets + extended evaluation.

---

## 5. Stage 2 — Reward-augmented fine-tuning ("RL")

**Important naming note:** This is not on-policy RL (no PPO/REINFORCE). Rewards are computed from decoded hypotheses (non-differentiable) and used to modulate per-sample CTC loss weights. The gradient flows through CTC only.

**Entry:** `run.sh` Stage 7 → `espnet2_train.py --task asr_rl --config conf/train_asr_rl.yaml`

**Model loading:** `init_param` points to the best SFT checkpoint (passed via `--init_param` by `run.sh`).

**Forward pass logic (summary, `RLESPnetModel.forward()` in `espnet2/asr/rl_espnet_model.py`):**

1. Compute standard CTC + attention loss (same as SFT path).
2. If `rl_weight > 0` and not every step (controlled by `reward_step_interval`): decode hypotheses under **`torch.no_grad()`** using CTC greedy decoding.
3. Compute **reward vector** (per utterance, values in ~[0,1]).
4. Apply **stage-2 objective** to combine CTC loss and rewards.
5. Log `reward_mean`, `reward_std`, and sample `(utt_id, reward, ref[:80], hyp[:80])` tuples periodically.

**Reward definitions:**

- **`mwer`:** per utterance, `reward = max(0.0, 1.0 - jiwer.wer(ref, hyp))`. Empty reference → 0.5 (neutral default).
- **`wwer`** (used in this run): per utterance, `reward = max(0.0, 1.0 - wwer_rate)` where `wwer_rate` is a **weighted WER** via dynamic-programming Levenshtein alignment over words:
  - Reference-word weights: `domain_term_weight` (3.0) if the word is in `domain_terms`, else 1.0.
  - Deletion and substitution costs use the reference-word weight. Insertion cost is always 1.0.
  - `wwer_rate = min(1.0, dp_distance / sum_of_ref_weights)` (reduces to standard WER when all weights = 1.0).
  - Empty reference → 0.5.
- **`llm`**: Gemini `gemini-1.5-flash` 0–1 quality score per pair. Mock mode (default): MWER + Gaussian noise (σ=0.02), clamped to [0,1]. Real mode: calls `google-generativeai` with `temperature=0.0`.
- **`all`**: arithmetic mean of `(mwer + wwer + llm) / 3.0`.

**Stage-2 objective options (`reward_loss_type`):**

- **`reweight_ctc` (used in this run; gradient-affecting):** compute per-sample CTC losses $\ell_i$, rewards $r_i \in [0,1]$, weights $w_i = 1 + \alpha(1-r_i)$ where $\alpha = \texttt{rl\_weight}$, and optimize:

$$L = \frac{1}{B}\sum_{i=1}^{B} w_i\,\ell_i$$

Higher-error utterances (lower reward) receive higher CTC loss weights, nudging the model to improve them.

- **`penalty` (legacy; not gradient-affecting through rewards):**

$$L = L_\text{CTC} + \alpha(1 - \bar{r})$$

**Reward caching + neutral fallback:** If `batch_idx % reward_step_interval != 0`, or if the long-utterance guard triggers (`encoder_len > max_encoder_len_for_reward`), the cached reward from the last compute step is reused. If no cached reward is available, a neutral constant of 0.5 is used.

**Hyperparameters:** Lower LR (`1e-5`) and fewer epochs (2) than SFT, FP32 precision, tighter gradient clipping (1.0) to avoid destabilizing the CTC fit while nudging toward higher reward.

**After RL:** Best checkpoint selected by `valid.acc`. Stage 8 runs decoding + extended evaluation on all sets.

---

## 6. Evaluation and analysis

**Decoding (Stage 8):** ESPnet2 greedy CTC decoding via `asr_inference.py`, producing hypothesis text files in `exp/asr_{sft,rl}/decode_<set>/`.

**Extended evaluation (`local/eval_extended.py`):**

Transcribes all sets and computes the following metrics (written to `extended_metrics.json`):

- **WER (%) / CER (%)**: `jiwer` library, multiplied by 100
- **SER (%)**: sentence error rate — fraction of utterances with at least one word error
- **EWER (%)**: "entity/domain WER" — WER restricted to domain-vocabulary tokens appearing in the reference; averaged over utterances that contain at least one domain token
- **Domain precision / recall / F1**: token-level metrics on occurrences of domain vocabulary terms
- **Degenerate hypothesis fraction**: fraction of hypotheses that are empty or repetitive
- **Mean hypothesis length (chars)**: average character length of decoded outputs
- **Paired bootstrap p-value** (`bootstrap_pval_vs_baseline`): two-sided approximate p-value for mean utterance-level WER difference between SFT and RL hypotheses on the same references (1000 bootstrap iterations)

**Forgetting evaluation:** LibriSpeech `dev_clean` is decoded by both the SFT and RL checkpoints to measure degradation on out-of-domain clean speech.

---

## 7. Key differences from NeMo implementation

| Dimension | NeMo | ESPnet2 (this recipe) |
|-----------|------|----------------------|
| **Framework** | PyTorch Lightning + NeMo `EncDecCTCModelBPE` | ESPnet2 custom trainer (`rl_trainer.py`) |
| **Loss** | CTC only (no attention decoder) | CTC + attention (joint; `ctc_weight=0.3`) |
| **Model** | `stt_en_conformer_ctc_medium` (~30M params, NGC) | `pyf98/librispeech_conformer` (HuggingFace) |
| **Data format** | NeMo JSONL manifests | Kaldi-style `wav.scp` / `text` |
| **Data combination** | AfriSpeech only (+ optional LibriSpeech forgetting) | AfriSpeech + VoxPopuli + LibriSpeech subset (combined training) |
| **Reward mode (run)** | MWER (WWER planned) | WWER (with 37 clinical domain terms) |
| **Reward weight** | 0.05 | 0.02 |
| **SFT precision** | FP32 | AMP (fp16) |
| **RL precision** | FP32 | FP32 |
| **Evaluation** | `model.transcribe(...)` (NeMo batch inference) | ESPnet2 `asr_inference.py` (CTC greedy) |
| **Metrics** | WER, CER, SER, EWER, domain P/R/F1, bootstrap p-value | Same + degenerate fraction, mean hyp length |

---

## 8. CLI quick reference (`run.sh`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--stage` / `--stop_stage` | `1` / `8` | Control which stages run |
| `--ngpu` | `1` | Number of GPUs; auto-falls-back to 0 if CUDA unavailable |
| `--train_set` | `train_combined` | Training data directory under `data/` |
| `--valid_set` | `afrispeech_dev` | Validation data directory |
| `--test_sets` | `afrispeech_test librispeech_dev_clean` | Evaluation sets |
| `--pretrained_model` | `pyf98/librispeech_conformer` | HuggingFace model ID for Stage 5 |
| `--seed` | `42` | Random seed passed to training |
| `--asr_config` | `conf/train_asr_sft.yaml` | SFT config (Stage 6) |
| `--rl_config` | `conf/train_asr_rl.yaml` | RL config (Stage 7) |

---

## 9. Robustness features

The recipe implements several guards to reduce manual triage:

- **CUDA availability guard:** If `--ngpu 1` but `torch.cuda.is_available()` is False, `run.sh` logs a warning and falls back to `ngpu=0` automatically.
- **Kaldi validation:** `utils/validate_data_dir.sh --no-feats` is called after each `data.sh` stage; empty `wav.scp` after Stage 2 raises a clear error.
- **HuggingFace streaming:** All dataset loads use `streaming=True` to avoid filling disk (the AfriSpeech-200 `all` config is 200+ hours).
- **Transcript sanitization:** Non-printable characters are stripped from HuggingFace transcripts before writing Kaldi text files.
- **Architecture auto-patching:** `local/setup_pretrained.py` reads the pretrained model config and patches `train_asr_sft.yaml` and `train_asr_rl.yaml` in-place to match architecture, normalization, and tokenizer paths.
- **Shape file generation:** Stage 4 generates shape files via an inline Python script, bypassing `asr_train.py --collect_stats` which rejects RL-specific config keys.
