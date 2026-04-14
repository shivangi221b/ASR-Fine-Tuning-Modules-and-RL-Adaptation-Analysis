# NVIDIA NeMo: project script and methodology

This document describes **this repository’s** NeMo training entrypoint (`nemo/gcp_scripts/nemo_afrispeech_training.py`): what it does end-to-end, how **supervised fine-tuning (SFT)** is configured, and how the **second stage** combines CTC loss with **utterance-level rewards** (the script labels this stage “RL”; mathematically it is **reward-augmented / regularized fine-tuning**, not a policy-gradient RL algorithm).

A short **NeMo ASR background** section at the end keeps high-level pointers to the generic stack (encoder–decoder, CTC, Lightning) for readers who are new to NeMo.

---

## 1. Script purpose and high-level flow

**File:** `nemo/gcp_scripts/nemo_afrispeech_training.py`

**Goal:** Domain adaptation for English CTC ASR using a **two-stage** procedure:

1. **Stage 1 — SFT:** Standard NeMo `EncDecCTCModelBPE` training with **CTC loss only** (no reward term).
2. **Stage 2 — Reward-augmented training:** Same model class, loaded from the SFT `.nemo` checkpoint, with **`training_step` monkey-patched** so the scalar loss becomes **CTC loss plus a small term derived from batch rewards** (WER-based, weighted-WER-based, optional LLM score, or an average of those).

**Default backbone:** `stt_en_conformer_ctc_medium` (overridable via `Config.NEMO_MODEL_NAME`; smoke tests use `stt_en_conformer_ctc_small`).

**Typical invocation:**

- Full pipeline (default): `python nemo_afrispeech_training.py` (or `--stage both`)
- SFT only: `--stage sft`
- Second stage only: `--stage rl --sft_checkpoint /path/to/sft_model.nemo`
- Quick sanity check: `--smoke_test`

**Artifacts:**

- Checkpoints: `CFG.CHECKPOINT_DIR` (default `./checkpoints`) — `sft_model.nemo`, `rl_model.nemo`
- Per-run metrics CSV: `CFG.RESULTS_DIR` (default `./results`) — SFT and RL epoch callback dumps
- Aggregated JSON: `CFG.RESULTS_DIR / {run_id}_results.json`
- Manifests and extracted audio live under the **repo** `data/manifests` and `data/audio` (paths are derived from the script location, not configurable in `Config`)

Optional **`gsutil`** upload of checkpoints, results JSON, and manifest directory when `UPLOAD_GCS_URI` / `--upload_gcs` is set (skipped in smoke test or if `SKIP_GCS`).

---

## 2. Configuration highlights (`Config`)

| Area | Defaults (non–smoke-test) | Notes |
|------|---------------------------|--------|
| **Datasets** | `afrispeech_clinical` | Alternatives: `librispeech`, `voxpopuli` |
| **Batch size** | 16 | CLI `--batch_size`; README suggests lowering on 16GB GPUs |
| **SFT** | `LEARNING_RATE_SFT=1e-4`, `SFT_EPOCHS=5` | AdamW + CosineAnnealing with warmup |
| **Stage 2** | `LEARNING_RATE_RL=1e-5` (10× lower than SFT), `RL_EPOCHS=2` | Comment in code: “1/10 SFT per paper plan” |
| **Reward** | `REWARD_MODE="mwer"`, `REWARD_WEIGHT=0.05`, `REWARD_STEP_INTERVAL=4` | Reward computed every N **batches**; skipped batches reuse cached rewards or 0.5 |
| **Long audio guard** | `MAX_ENCODER_LEN_FOR_REWARD=2000` | If max frame length in batch exceeds this, reward forward/decode is skipped; cached or neutral rewards used |
| **LoRA** | `USE_LORA=False` | `--use_lora` tries NeMo `add_adapter` with a linear adapter config (version-dependent) |
| **LLM reward** | `GEMINI_MODEL="gemini-1.5-flash"`, `USE_MOCK_LLM=True` | `--real_llm` sets live Gemini when `GEMINI_API_KEY` is set; otherwise MWER-based fallback |
| **Eval / analysis flags** | Zero-shot val, LibriSpeech forgetting eval, final test eval, bootstrap iters | Toggle via `skip_*` CLI flags |

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

**Trainer (Lightning):** single device, GPU if available else CPU, `precision="16-mixed"` on CUDA else `32-true`, `max_epochs=SFT_EPOCHS`, **no Lightning checkpointing** (`enable_checkpointing=False`), no default logger; custom **`NemoTrainingLogger`** appends train/val callback metrics to CSV each validation epoch end.

**After SFT:** `model.save_to(save_path)` (`.nemo`), optional GCS upload, **`evaluate_manifest_bundle`** on the validation manifest (WER, CER, SER, domain-centric metrics — see §6).

---

## 5. Stage 2 — Reward-augmented training (“RL” in logs)

**Important naming note:** The code and logs refer to this as **RL**, but the implementation is **not** REINFORCE / PPO / actor–critic. It keeps the **CTC objective** and adds a **scalar batch penalty** derived from **non-differentiable** decode-and-score rewards. Gradients flow only through **CTC**; rewards shape training **indirectly** by shifting the loss surface batch-to-batch (similar in spirit to auxiliary criteria used in some ASR “MWER training” discussions, but here the auxiliary term is **`reward_weight * (1 - mean reward)`**, not a full minimum-Bayes-risk gradient).

**Entry:** `run_rl_stage` → `load_model_for_rl` → `trainer.fit(model)`.

**Model loading:** `EncDecCTCModelBPE.restore_from(sft_checkpoint)`; sets `reward_mode`, `reward_weight`, clears logs, optionally attaches adapter.

**Patch:** `original_training_step = model.training_step`, then `model.training_step = types.MethodType(patched_training_step, model)`.

**Patched step logic (summary):**

1. Call **original** `training_step` to get `ctc_loss` (dict or tensor).
2. Extract batch tensors (supports multiple NeMo batch layouts: `.audio` / `.input_signal`, etc.).
3. **Long-audio batch:** if `signal_len.max() > MAX_ENCODER_LEN_FOR_REWARD`, skip expensive decode; use **cached** prior batch rewards or **0.5** per utterance.
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

7. **Loss:** `penalty = 1.0 - rewards.mean()`; **`total_loss = ctc_loss + reward_weight * penalty`**. Replace `result["loss"]` with `total_loss`.
8. Append step diagnostics to `model._step_logs` (CTC loss, reward mean, penalty, flags for long audio / skipped reward).

**Hyperparameters:** Lower LR (`1e-5`) and fewer epochs (`2`) than SFT to avoid destroying the CTC fit while nudging toward higher reward.

**After stage 2:** Save `rl_model.nemo`, validate, attach **`reward_trajectory`** (mean reward per logged step) and summary stats to results.

---

## 6. Evaluation and analysis built into the pipeline

**`evaluate_manifest_bundle`:** Transcribe manifest with `model.transcribe`, then:

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
