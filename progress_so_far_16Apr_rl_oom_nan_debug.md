# Progress log (16 Apr 2026): RL-stage NaNs/OOM investigation and fixes

This note documents a focused debugging session on **stage-2 (“RL”)** for AfriSpeech clinical runs in `nemo/gcp_scripts/nemo_afrispeech_training.py`, specifically:

- RL collapse / NaNs observed in `afrispeech_clinical_seed42_1776207077` (WWER reward).
- A reproducible **CUDA OOM** encountered when running RL-only with debug enabled on a ~16GB GPU.
- The resulting code refactor to eliminate **multiple forward passes per batch** (root cause of peak VRAM), while keeping **strict non-finite checks** for publishable runs.

Paper-facing methodology remains in `nemo/docs/nemo_analysis.md` (no debugging narrative).

---

## 1) Symptom summary (what was wrong)

### 1.1 Full run RL collapse in `afrispeech_clinical_seed42_1776207077`

From the run JSON (`vm_results/..._results.json`):

- **SFT**: reasonable (`wer≈45.95`, `_degenerate_hyp_frac≈0.00055`)
- **RL**: catastrophic collapse (`wer=100`, `cer=100`, `_degenerate_hyp_frac=1.0`, mean hyp len ≈ 1 char)
- RL epoch CSV showed `val_loss = nan` during/after RL.

From `debug/debug_samples_rl.jsonl`:

- RL hypotheses were initially plausible but degraded step-by-step until they became almost entirely the `⁇` glyph.

Interpretation:

- This pattern is consistent with RL training producing **non-finite values** (NaN/Inf) at some point, which then corrupts gradients/weights and causes decoding collapse.

---

## 2) “No shortcuts” requirement and diagnostic strategy

We explicitly avoided a “silent fallback” that would hide numerical issues.

### 2.1 Strict non-finite checks (publishable behavior)

RL `training_step` was modified to **hard-fail** (raise) on first occurrence of:

- non-finite `log_probs` (from the model forward),
- non-finite per-sample `ctc_per`,
- non-finite reward vector or rewards outside \([0,1]\),
- non-finite `total_loss`.

Diagnostics include tensor stats (shape/dtype/device/min/max/mean/finite fraction) and CTC length details (input lengths, target lengths, blank id, targets_flat length).

Outcome:

- This turns “mysterious collapse later” into a **precise first-failure trace**.

---

## 3) New failure discovered: CUDA OOM (GPU memory), not disk

When attempting RL-only runs with debug enabled, we hit:

- `torch.OutOfMemoryError: CUDA out of memory ...` inside Conformer attention (`rel_shift`).

Key clarification:

- Deleting logs or old results does **not** help CUDA OOM.
- CUDA OOM is **GPU VRAM** pressure; the fix is to reduce per-step VRAM peak (batch size, sequence lengths, or computation graph size), and to restart the Python process after OOM.

---

## 4) Root cause of OOM: multiple forward passes per batch in RL

Before the refactor, RL could execute up to **three** forward passes per batch:

1. One forward via NeMo’s native `training_step` (CTC loss).
2. Another forward under `torch.no_grad()` to decode hypotheses and compute reward.
3. A third forward to compute per-sample `ctc_per` for `reweight_ctc`.

Even if step (2) uses `no_grad`, it still runs large activations; with long utterances and batch size 16 this pushed VRAM over the limit.

---

## 5) Code change: single-forward RL training step (VRAM fix)

We refactored RL `patched_training_step` to use **exactly one model forward** per batch:

- Compute `log_probs_g, encoded_len_g = self.forward(...)` once.
- Compute per-sample `ctc_per = F.ctc_loss(...)` from `log_probs_g` (with gradients).
- If `compute_now=True` (and not long_audio), compute reward by decoding from **detached logits**:
  - `decoder_outputs = log_probs_g.detach()`
  - `decoder_lengths = encoded_len_g.detach()`
  - then compute MWER/WWER/LLM reward from decoded hyps vs refs.

This preserves:

- the `reweight_ctc` objective (reward-weighted per-sample CTC),
- strict non-finite checks,
- reward caching / step-interval behavior,

while removing peak VRAM spikes from extra forward passes.

---

## 6) Runtime mitigation: reduce batch size to avoid VRAM peaks

Even with single-forward, long utterances can cause VRAM spikes. For ~16GB GPUs:

- recommended RL batch size: **8** (and **4** if still OOM).

Optional allocator tweak for fragmentation:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 7) Smoke test vs full RL run

RL-only **smoke tests** are still important before launching full RL runs:

- `--smoke_test` shrinks AfriSpeech train/val/test caps and epochs.
- It is used to validate that reward computation runs, dumps/checkpoints write correctly, and NaN checks do not trip immediately.

Once smoke is stable, run RL-only on a medium slice (e.g., 2000/300) to reproduce longer-horizon issues without paying full-run cost.

---

## 8) What to report (and what not to)

Safe statements:

- “We modified the stage-2 implementation to avoid multiple forward passes per batch by decoding rewards from detached logits computed in the same forward pass used for CTC loss.”
- “We reduced batch size for stage 2 on memory-limited GPUs to avoid OOM.”

Do **not** claim improvements from runs where RL is NaN/collapsed.

