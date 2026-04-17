# NVIDIA NeMo: experiment driver + methods (paper-facing)

This document is the **paper-facing methods write-up** for this repository’s NeMo ASR experiment driver: `nemo/gcp_scripts/nemo_afrispeech_training.py`.

It covers the **two-stage methodology** (SFT → reward-augmented fine-tuning), the **reward definitions and objectives**, and the **hyperparameters** that define each run.

It intentionally **does not** include debugging narrative. (See `progress_so_far_14Apr.md` for the full debug timeline and fixes validation.)

---

## 1. Purpose and high-level flow

**File:** `nemo/gcp_scripts/nemo_afrispeech_training.py`

**Goal:** Domain adaptation for English **CTC-BPE** ASR using a **two-stage** procedure:

1. **Stage 1 — SFT:** Standard NeMo `EncDecCTCModelBPE` training with **CTC loss only** (no reward term).
2. **Stage 2 — Reward-augmented fine-tuning (“RL”):** Load the SFT `.nemo` checkpoint and fine-tune with a reward-driven objective. Rewards are computed from decoded hypotheses (non-differentiable) and incorporated into the stage-2 objective (see §5).

**Default backbone:** `stt_en_conformer_ctc_medium` (overridable via `Config.NEMO_MODEL_NAME`; smoke tests use `stt_en_conformer_ctc_small`).

**Typical invocation:**

- Full pipeline (default): `python nemo_afrispeech_training.py` (or `--stage both`)
- SFT only: `--stage sft`
- Second stage only: `--stage rl --sft_checkpoint /path/to/sft_model.nemo`
- Quick sanity check: `--smoke_test`

**Artifacts (per run):**

- Checkpoints: `CFG.CHECKPOINT_DIR` (default `./checkpoints`) — `sft_model.nemo`, `rl_model.nemo`
- Per-run metrics CSV: `results/<run_id>/..._{sft,rl}_epoch_metrics.csv`
- Aggregated JSON: `results/<run_id>/<run_id>_results.json`
- Lightning resume checkpoints: `results/<run_id>/checkpoints/{sft,rl}/last.ckpt`
- Stage-local exports: `results/<run_id>/exports/{sft_model.nemo,rl_model.nemo}`
- Manifests and extracted audio live under the **repo** `data/manifests` and `data/audio` (paths are derived from the script location, not configurable in `Config`)

Optional **`gsutil`** upload of checkpoints, results JSON, and manifest directory when `UPLOAD_GCS_URI` / `--upload_gcs` is set (skipped in smoke test or if `SKIP_GCS`).

---

## 1.1 What runs when (AfriSpeech vs LibriSpeech) and sample counts

When `CFG.DATASET=afrispeech_clinical` (the default), **training and primary evaluation are on AfriSpeech**.

LibriSpeech is used only for an **optional “catastrophic forgetting” evaluation**:

- It does **not** participate in training for the AfriSpeech run.
- It is evaluated **twice** in the full pipeline when enabled:
  - **after SFT** (`results["librispeech_after_sft"]`)
  - **after stage 2 / “RL”** (`results["librispeech_after_rl"]`)

**Smoke test (`--smoke_test`) sample counts (AfriSpeech clinical):**

- Train: `TRAIN_SAMPLES=30`
- Val: `VAL_SAMPLES=10`
- Test: `TEST_SAMPLES=5`
- LibriSpeech forgetting eval: **disabled** by default in smoke test (`RUN_LIBRISPEECH_FORGETTING=False`)

**Full (non-smoke) sample counts (AfriSpeech clinical):**

- Train: `TRAIN_SAMPLES=None` → all available clinical train utterances (after filtering)
- Val: `VAL_SAMPLES=None` → all available clinical validation utterances
- Test: `TEST_SAMPLES=None` → all available clinical test utterances
- LibriSpeech forgetting eval: enabled by default (`RUN_LIBRISPEECH_FORGETTING=True`) unless `--skip_librispeech_forgetting`
  - LibriSpeech val subset size: **2,703 utterances**

## 2. Configuration and hyperparameters (`Config`)

| Area | Defaults (non–smoke-test) | Notes |
|------|---------------------------|--------|
| **Model** | `NEMO_MODEL_NAME=stt_en_conformer_ctc_medium` | Smoke tests use `stt_en_conformer_ctc_small` |
| **Datasets** | `DATASET=afrispeech_clinical` | Alternatives: `librispeech`, `voxpopuli` |
| **Batch size** | `BATCH_SIZE=16` | CLI `--batch_size` |
| **SFT** | `LEARNING_RATE_SFT=1e-4`, `SFT_EPOCHS=5` | AdamW + CosineAnnealing w/ warmup |
| **Stage 2 (“RL”)** | `LEARNING_RATE_RL=1e-5`, `RL_EPOCHS=2` | “1/10 SFT LR” heuristic |
| **Reward mode** | `REWARD_MODE=mwer` | `mwer` / `wwer` / `llm` / `all` |
| **Reward weight** | `REWARD_WEIGHT=0.05` | Scales stage-2 objective |
| **Reward interval** | `REWARD_STEP_INTERVAL=4` | Compute reward every N batches; reuse cached otherwise |
| **Reward long-audio guard** | `MAX_AUDIO_SECONDS_FOR_REWARD=25.0`, `SAMPLE_RATE=16000` | Skip reward compute when utterances are too long |
| **Stage-2 objective** | `RL_OBJECTIVE=reweight_ctc` | `reweight_ctc` (gradient-affecting) or `add_penalty` (legacy) |
| **Precision** | `FORCE_FP32=True` | Lightning `precision="32-true"` by default |
| **Gradient clipping** | `GRAD_CLIP_VAL=1.0` | Lightning `gradient_clip_val` |
| **Text normalization** | `NORMALIZE_TEXT=False` | `--normalize_text` rewrites manifests in `results/normalized_manifests/` |
| **LoRA** | `USE_LORA=False` | `--use_lora` best-effort adapter attach |
| **LLM reward** | `GEMINI_MODEL=gemini-1.5-flash`, `USE_MOCK_LLM=True` | `--real_llm` uses Gemini if `GEMINI_API_KEY` |
| **Eval flags** | Zero-shot, forgetting eval, test eval, bootstrap iters | Toggle via `skip_*` flags |

Smoke test (`apply_smoke_test_overrides`) shrinks data caps, epochs, batch size, disables Libri forgetting eval by default, uses small model, skips GCS, reduces bootstrap iterations.

---

## 3. Data preparation and manifests

The script does **not** assume pre-built NeMo manifests for AfriSpeech; it calls into the repo **`data`** package:

- **`afrispeech_clinical`:** `prepare_afrispeech_clinical_manifests_streaming(...)` → train/val/(optional) test JSONL paths under `data/manifests`, audio under `data/audio`.
- **`librispeech` / `voxpopuli`:** `load_dataset_bundle` then `build_nemo_manifest` per split; audio under `data/audio/<dataset_name>/` to avoid filename collisions.

**NeMo dataloader config** (`build_data_config`): 16 kHz, `max_duration` 20 s, `min_duration` 0.5 s on train, `trim_silence=False`, bucketing handled by NeMo defaults on the config dict. Worker count is 0 in smoke test, else 4.

---

## 4. Stage 1 — Supervised fine-tuning (SFT)

**Entry:** `run_sft_stage` → `load_model_for_sft` → `trainer.fit(model)`.

**Model loading:** `EncDecCTCModelBPE.from_pretrained(CFG.NEMO_MODEL_NAME)` (NGC/HF-style NeMo pretrained id).

**Reward disabled for SFT:** `model.reward_weight = 0.0`; internal logging slots (`_step_logs`, `_reward_batch_idx`, `_cached_batch_reward`) initialized. No patch to `training_step` — **pure NeMo CTC training**.

**Optimization:** `model.setup_optimization(optim)` with OmegaConf dict:

- Optimizer: **AdamW**, `lr=LEARNING_RATE_SFT`, `weight_decay=1e-3`
- Scheduler: **CosineAnnealing** with `warmup_steps` from `compute_warmup_steps` (~10% of total steps, capped)

**Trainer (Lightning):** single device, GPU if available else CPU, fp32 by default (`FORCE_FP32=True` → `precision="32-true"`), gradient clipping via `GRAD_CLIP_VAL`, per-epoch Lightning checkpoints under `results/<run_id>/checkpoints/sft/`, and CSV metrics via `NemoTrainingLogger`.

**After SFT:** `model.save_to(save_path)` (`.nemo`), optional GCS upload, **`evaluate_manifest_bundle`** on the validation manifest (WER, CER, SER, domain-centric metrics — see §6).

---

## 5. Stage 2 — Reward-augmented fine-tuning (“RL” in logs)

**Important naming note:** This is not on-policy RL (no PPO/REINFORCE). Rewards are computed from decoded hypotheses (non-differentiable) and used to modify the stage-2 objective.

**Entry:** `run_rl_stage` → `load_model_for_rl` → `trainer.fit(model)`.

**Model loading:** `EncDecCTCModelBPE.restore_from(sft_checkpoint)`; sets `reward_mode`, `reward_weight`, clears logs, optionally attaches adapter.

**Patch:** `original_training_step = model.training_step`, then `model.training_step = types.MethodType(patched_training_step, model)`.

**Patched step logic (summary):**

1. Call **original** `training_step` to get `ctc_loss` (dict or tensor).
2. Extract batch tensors (supports multiple NeMo batch layouts: `.audio` / `.input_signal`, etc.).
3. **Long-audio batch:** if `max(signal_len)/SAMPLE_RATE > MAX_AUDIO_SECONDS_FOR_REWARD`, skip expensive reward decode/score; use cached rewards or neutral 0.5.
4. **Step interval:** if `batch_idx % REWARD_STEP_INTERVAL != 0`, reuse **cached** rewards or 0.5.
5. Otherwise, under **`torch.no_grad()`**:
   - Forward: `log_probs, encoded_len, _ = self.forward(input_signal=signal, input_signal_length=signal_len)`
   - Greedy-like CTC predictions: `self.wer.decoding.ctc_decoder_predictions_tensor(...)`
   - References from token ids via `self.tokenizer.ids_to_text`
6. **Reward vector** (per utterance, values in ~[0,1]):

   - **`mwer`:** `1 - jiwer.wer(ref, hyp)` (clamped)
   - **`wwer`:** `1 - weighted_wer_rate(...)` with domain term weights (clinical vs parliamentary set by `DATASET`)
   - **`llm`:** Gemini 0–1 score from a short prompt, per pair; on failure / no key / mock: **MWER with small noise** or plain MWER
   - **`all`:** average of MWER, WWER, LLM rewards

7. **Stage-2 objective:** controlled by `RL_OBJECTIVE`.
8. Append step diagnostics to `model._step_logs` (CTC loss, reward mean, penalty/surrogate, flags).

**Stage-2 objective options (`RL_OBJECTIVE` / `--rl_objective`):**

- **`reweight_ctc` (default; gradient-affecting):** compute per-sample CTC losses \(\ell_i\), rewards \(r_i \in [0,1]\), weights \(w_i = 1 + \alpha(1-r_i)\) where \(\alpha=\texttt{REWARD_WEIGHT}\), and optimize

\[
L = \frac{1}{B}\sum_{i=1}^{B} w_i\,\ell_i
\]

- **`add_penalty` (legacy; not gradient-affecting):**

\[
L = \mathrm{CTC} + \alpha(1-\overline{r})
\]

**Hyperparameters:** Lower LR (`1e-5`) and fewer epochs (`2`) than SFT to avoid destroying the CTC fit while nudging toward higher reward.

**After stage 2:** Save `rl_model.nemo`, validate, attach **`reward_trajectory`** (mean reward per logged step) and summary stats to results.

---

## 6. Evaluation and analysis built into the pipeline

**`evaluate_manifest_bundle`:** Transcribe manifest with `model.transcribe` under `eval()` + `torch.no_grad()`, then:

- **WER / CER / SER** (jiwer; SER = exact normalized sentence match rate)
- **EWER** (`entity_wer_from_text`): WER computed on **domain vocabulary tokens** appearing in the reference
- **Domain term precision / recall / F1** over domain lexicon occurrences

**`run_full_pipeline` additionally:**

- **`zero_shot_val`:** Fresh pretrained model on val manifest (if `RUN_ZERO_SHOT`)
- **Catastrophic forgetting proxy:** LibriSpeech val manifest (`prepare_librispeech_eval_manifest`) after SFT and after RL (`RUN_LIBRISPEECH_FORGETTING`)
- **Paired bootstrap p-value** (`paired_bootstrap_wer_pvalue`): two-sided approximate p-value for mean **utterance-level WER** difference between **SFT vs RL** hypotheses on the **same** references (`BOOTSTRAP_ITERS`, default 1000)
- **Held-out test** (AfriSpeech): `test_sft` and `test_rl` if test manifest exists and `RUN_FINAL_TEST_EVAL`

Results are merged into one JSON; large `_refs` / `_hyps` fields are stripped before save.

---

## 7. CLI quick reference

| Flag | Effect |
|------|--------|
| `--smoke_test` | Tiny data, 1 epoch stages, small model, skip GCS / most forgetting |
| `--stage {sft,rl,both}` | Run subset of pipeline |
| `--sft_checkpoint` | Required for `--stage rl` |
| `--dataset {afrispeech_clinical,librispeech,voxpopuli}` | Data domain |
| `--train_samples`, `--val_samples`, `--test_samples` | Caps for AfriSpeech clinical |
| `--voxpopuli_train_subset` | Train cap for VoxPopuli |
| `--reward_mode`, `--reward_weight` | Override `REWARD_MODE` / `REWARD_WEIGHT` |
| `--rl_objective {reweight_ctc,add_penalty}` | Choose stage-2 objective |
| `--force_fp32` | Force fp32 (default is fp32) |
| `--grad_clip_val` | Set gradient clipping |
| `--normalize_text` | Rewrite manifests with conservative normalization |
| `--upload_gcs gs://...` | Upload artifacts |
| `--use_lora` | Attempt encoder adapter attach |
| `--mock_llm` / `--real_llm` | LLM reward behavior |
| `--skip_zero_shot`, `--skip_librispeech_forgetting`, `--skip_test_eval` | Disable optional eval branches |
| `--batch_size` | Override batch size (ignored for batch size if smoke test path—smoke sets its own) |

---

## 8. NeMo ASR background (generic reference)

This is **not** specific to the script but helps situate the implementation.

**Pipeline structure:** NeMo ASR commonly uses **Conformer / FastConformer** encoders with **CTC** or **RNN-T** heads; configs live in NeMo’s `examples/asr/conf`. This project uses **CTC BPE** (`EncDecCTCModelBPE`).

**Typical stack:** Manifest JSONL (audio path + text) → NeMo collection dataloaders → Lightning `Trainer` → CTC loss + optional auxiliary losses (e.g. InterCTC in other recipes).

**PEFT in NeMo:** NeMo 2.x emphasizes **model transforms / adapters** for many models; this script’s `maybe_attach_lora` path is a **pragmatic, version-tolerant** attempt to add a **linear adapter** on the encoder, not a guarantee across all NeMo versions.

**True RL / MWER-style objectives in NeMo:** Replacing or augmenting loss with **full** risk-based objectives usually requires **custom model subclasses** or training loops; this script stays within **Lightning + patched `training_step`** and keeps **CTC** as the only direct gradient source from the decoder.

---

## 9. Relation to older versions of this doc

Earlier revisions of `nemo_analysis.md` focused on **generic** NeMo components, `speech_to_text_finetune.py`, and hypothetical RL hooks. They have been **replaced / superseded** by §§1–7 above for **this repository’s** experiment driver; §8 retains a condensed generic reference where it still helps.
