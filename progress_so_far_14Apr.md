# Progress so far (14 Apr 2026): NeMo SFT + reward-augmented fine-tuning (“RL”)

This note is the **comprehensive progress + debugging log** for the NeMo experiment driver (`nemo/gcp_scripts/nemo_afrispeech_training.py`): what worked, what failed, what evidence we collected, and what exact code changes fixed each issue.

Paper-facing methodology (clean, no debugging narrative) lives in `nemo/docs/nemo_analysis.md`.

---

## Key runs referenced

- **Initial full run (broken):** `results/results/afrispeech_clinical_seed33_1776117737_results.json` (SFT/RL WER=100, RL reward trajectory flat 0.5)
- **GCP smoke runs (debug):** multiple `results/afrispeech_clinical_seed42_*` folders; the critical “fixed” run is:
  - `results/afrispeech_clinical_seed42_1776199250/afrispeech_clinical_seed42_1776199250_results.json`
  - plus its debug dumps and epoch CSVs under that folder

---

## 1) What worked early (even before fixes)

### 1.1 The experiment configuration is coherent and paper-ready

From `..._results.json`, the run configuration is internally consistent:

- **Backbone**: `stt_en_conformer_ctc_medium` (NeMo Conformer CTC BPE)
- **Dataset**: `afrispeech_clinical`
- **Seed**: 33
- **Stage 1 (SFT)**: AdamW + cosine schedule, LR \(1e-4\), 5 epochs, batch size 16
- **Stage 2 (“RL”)**: LR \(1e-5\), 2 epochs, reward mode `mwer`, reward weight 0.05, reward interval every 4 steps
- **Evaluation toggles**: zero-shot val, forgetting eval (LibriSpeech), and final test eval were enabled
- **Run bookkeeping**: includes GCS upload prefix `gs://adaptive-ai-487419-stt-results/afrispeech_20260413_2118`

These items can be used in a survey/paper as a clear “recipe” even though the adapted-model metrics are currently invalid (see §2).

### 1.2 Zero-shot baseline results *do* look plausible (and why)

The **zero-shot validation** metrics on AfriSpeech clinical look reasonable for a domain-mismatched baseline:

- **WER** ≈ 57.88%
- **CER** ≈ 25.87%
- **SER** = 100% (no exact sentence matches; not surprising with high WER)
- **n_utterances** = 1813

Additionally, the script reports domain-term precision/recall/F1 and EWER (domain-token-focused error), which can be described as **“domain lexicon sensitivity”** metrics. These particular domain-term scores are relatively high in the zero-shot output; interpret cautiously (they can be inflated if the model frequently emits common clinical tokens even when the full sentence is incorrect).

**How to use this in the survey paper:**

- Present this as the **starting point**: a strong pretrained NeMo English CTC baseline on AfriSpeech clinical.
- Emphasize that **domain shift remains large** (high WER), motivating adaptation.
- Mention that the pipeline supports both **standard ASR metrics** (WER/CER) and **domain-centric metrics** (EWER, domain-term F1) to better capture clinical vocabulary correctness.

### 1.3 The pipeline successfully generates end-to-end artifacts

Even though the SFT/RL model quality is currently broken, the run did:

- Produce a **single JSON bundle** with config + all evaluation sections.
- Produce stage-specific **epoch metric CSVs** (though the original CSV header format was broken; fixed in code—see §3.2).
- Record per-stage wall-clock training times:
  - SFT train time ~ 2.36 hours (`train_time_s ≈ 8484`)
  - RL train time ~ 0.94 hours (`train_time_s ≈ 3395`)

This is useful operationally for paper planning (budgeting and describing experimental procedure), once metrics are corrected.

---

## 2) What was going wrong (root causes + evidence)

### 2.1 Post-SFT and post-stage-2 metrics are “all 100% error”

The JSON reports **WER=100, CER=100, SER=100** on:

- AfriSpeech validation (after SFT and after stage 2)
- LibriSpeech forgetting evaluation (after SFT and after stage 2)
- AfriSpeech test (after SFT and after stage 2)

In practice, this pattern most often happens when hypotheses become **empty strings** (or a degenerate constant output) for most utterances, which yields near-maximal error under WER/CER.

**Why this is a strong red flag:**

- The same run’s **zero-shot** WER is ~58%, which is plausible.
- It is extremely unlikely that SFT training on AfriSpeech would reliably degrade performance to a uniform 100% error across *every* evaluation set unless something is broken (either training collapse or evaluation producing empty/invalid hypotheses).

### 2.2 Stage-2 reward trajectory flat at 0.5 (reward never computed)

In the JSON:

- `reward_trajectory`: all values are exactly **0.5**
- `reward_mean = 0.5`, `reward_std = 0.0`

In the script’s implementation, **0.5 is the fallback “neutral reward”** used when reward computation is skipped (or no cached reward exists). A completely flat 0.5 strongly implies that reward computation was **always skipped**.

### 2.3 Root cause: incorrect “long audio” guard (samples mistaken for frames)

Originally, stage-2 reward computation was skipped when:

- `signal_len.max() > MAX_ENCODER_LEN_FOR_REWARD` with `MAX_ENCODER_LEN_FOR_REWARD=2000`

But in NeMo batches, `signal_len` is typically **audio samples**. At 16 kHz, 2000 samples ≈ 0.125 seconds. That means essentially every real utterance is considered “long,” so the reward path never runs → always fallback 0.5 → stage 2 provides no real reward shaping.

This explains the **flat reward trajectory**.

### 2.4 CSV logging format problem (secondary, but affects analysis)

---

## 3) Debugging timeline (what we tried, what we learned, and what fixed it)

### 3.1 Added observability: sample dumps, resume checkpoints, and debug logging

**Change:** Added persistent artifacts so we could debug without guessing:

- **Sample dumps** (`--debug_sample_dump`): JSONL files under `results/<run_id>/debug/` that store `audio_filepath`, `reference_text`, and `hypothesis_text` for a fixed set of validation examples, plus summary headers per dump.
- **Per-epoch Lightning checkpoints**: `results/<run_id>/checkpoints/{sft,rl}/last.ckpt` so runs can resume.
- **Reward debug logs** (`--debug_reward`): logs skip reasons, max audio length in seconds, and empty-hyp stats.
- **CSV fix**: epoch metrics CSV is now parseable (header rewritten if metrics keys change).

**Outcome:** We could prove when hypotheses were degenerate and whether reward computation ran.

### 3.2 Fixed Stage-2 reward always being skipped

**Symptom:** Stage-2 `reward_trajectory` was exactly 0.5 for all steps.

**Evidence:** GCP logs showed `long_audio=True` nearly always with the original threshold.

**Fix:** Replaced the guard based on `signal_len` with a seconds-based guard:

- `MAX_AUDIO_SECONDS_FOR_REWARD` (default 25s; smoke tests set to 30s)
- `long_audio = (signal_len.max()/SAMPLE_RATE) > MAX_AUDIO_SECONDS_FOR_REWARD`

**Outcome:** Reward computation began to run; reward trajectory became non-constant in later smoke tests.

### 3.3 Reproduced “WER=100” and identified the true failure mode (`⁇` collapse)

**Symptom:** After SFT, WER/CER/SER became 100%.

**Evidence:** `debug_samples_sft.jsonl` showed that at step 0 hypotheses were normal, but by **train_step_5** all hypotheses became the special unknown glyph `⁇` (not empty strings).

**Fixes for observability (not the root fix):**

- Forced transcription-based evaluation to run under `model.eval()` + `torch.no_grad()`.
- Added `degenerate_hyp_frac` which treats “only `⁇`” as degenerate (previous “empty” metric missed this).

**Conclusion:** The model was collapsing numerically, not merely returning empty strings.

### 3.4 Ruled out tokenizer UNK pressure as the cause

We added tokenizer UNK diagnostics:

- train `mean_unk_frac ≈ 0.007`, val `mean_unk_frac ≈ 0.014` (very low)

**Conclusion:** collapse to `⁇` was not driven by targets mapping to UNK.

### 3.5 Found the real root cause: AMP + NaNs during/after SFT

**Evidence:** SFT epoch CSV for the broken run showed `val_loss = nan`.

This aligned with the observed behavior: after a few optimizer steps, decoding collapses to `⁇`.

**Fix:** Make training numerically stable by default:

- Set `FORCE_FP32=True` by default (Lightning `precision="32-true"`).
- Enable gradient clipping by default (`GRAD_CLIP_VAL=1.0`).

**Outcome (verified):** Smoke test run `afrispeech_clinical_seed42_1776199250` produced finite losses and non-degenerate outputs:

- `sft.wer ≈ 37.66`, `_degenerate_hyp_frac = 0.0`
- `rl.wer ≈ 44.16`, `_degenerate_hyp_frac = 0.0`
- reward trajectory non-constant, and decoding remained normal over steps.

### 3.6 Upgraded stage-2 objective to be “meaningful”

**Issue:** The earlier stage-2 loss \(CTC + \alpha(1-\bar{r})\) adds a constant penalty and does not change gradients beyond CTC.

**Fix:** Implement `RL_OBJECTIVE=reweight_ctc` (default), which computes per-sample CTC loss and reweights gradients based on reward:

\[
L = \frac{1}{B} \sum_i \big(1 + \alpha(1-r_i)\big)\,\ell_i
\]

This makes stage 2 actually incorporate reward into gradient magnitudes.
---

## 4) Current status (as of 14 Apr 2026 end)

- **Reward computation**: working and observable.
- **Training stability**: fp32 + gradient clipping prevents NaNs and prevents `⁇` collapse.
- **Stage-2 objective**: upgraded to gradient-affecting `reweight_ctc`.
- **Artifacts**: per-run checkpoints + exports + debug dumps are generated and (optionally) uploaded to GCS.
---

## 5) What to report in the paper right now vs later

### 5.1 Safe to report now

- The full experimental protocol (SFT → reward-augmented fine-tuning), datasets, and metrics suite.
- Zero-shot baselines.
- For small-scale sanity checks, report results only with enough validation data to reduce noise (≥200 utterances).

### 5.2 Hold until larger-scale runs complete

- Any claims that stage-2 improves overall WER: small slices are noisy; stage-2 may trade off WER vs domain-term behavior.
- Statistical significance: use paired bootstrap on a substantial validation set.

