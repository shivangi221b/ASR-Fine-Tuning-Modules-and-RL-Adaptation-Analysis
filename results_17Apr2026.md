# Results summary — 17 April 2026

This note compiles **AfriSpeech clinical** (and related) metrics from two `vm_results` folders. Tables below use HTML with **inline** wrap-friendly styles (works in Cursor preview and many viewers; pipe tables stay on one long line).

<table style="width:100%; border-collapse:collapse; table-layout:fixed; margin:0.75em 0; font-size:0.95em;">
<thead>
<tr>
<th style="width:14%; border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere; background:#f6f8fa; text-align:left;">Role</th>
<th style="width:26%; border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere; background:#f6f8fa; text-align:left;"><code>vm_results</code> folder</th>
<th style="width:22%; border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere; background:#f6f8fa; text-align:left;"><code>run_id</code></th>
<th style="width:38%; border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere; background:#f6f8fa; text-align:left;">Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;"><strong>SFT + full pipeline config</strong></td>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;"><code>vm_results/sft_working_afrispeech_clinical_seed42_1776207077/</code></td>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;"><code>afrispeech_clinical_seed42_1776207077</code></td>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;">Complete <code>*_results.json</code>: <code>config</code>, zero-shot, SFT, Libri after SFT, <strong>collapsed RL</strong> (invalid), test splits, bootstrap <code>p</code>-value.</td>
</tr>
<tr>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;"><strong>Healthy RL-only artifact</strong></td>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;"><code>vm_results/afrispeech_clinical_seed42_rl_1776462369/</code></td>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;"><code>afrispeech_clinical_seed42_rl_1776462369</code></td>
<td style="border:1px solid #ccc; padding:8px 10px; vertical-align:top; word-wrap:break-word; overflow-wrap:anywhere;">JSON has <strong>RL test metrics only</strong> (no embedded <code>config</code>, no SFT block). Assumes <strong>same training recipe</strong> as config in <code>1776207077</code> unless the VM job overrode flags.</td>
</tr>
</tbody>
</table>

**Paper-facing caveat:** ideal reporting uses **one** `*_results.json` from a single end-to-end run (SFT → RL → eval). Here, **SFT numbers come from `1776207077`** and **AfriSpeech RL numbers come from `1776462369`**. Treat LibriSpeech-after-RL, `test_rl`, and bootstrap p-value in `1776207077` as tied to the **failed** RL in that file, not to `1776462369`.

---

## 1) Hyperparameters and run settings (from `1776207077` `config`)

<table style="width:100%;border-collapse:collapse;table-layout:fixed;margin:0.75em 0;">
<thead><tr><th style="width:32%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Field</th><th style="width:68%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Value</th></tr></thead>
<tbody>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Base model</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>stt_en_conformer_ctc_medium</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Dataset</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>afrispeech_clinical</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Train / val / test caps</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>TRAIN_SAMPLES</code>, <code>VAL_SAMPLES</code>, <code>TEST_SAMPLES</code> = <code>null</code> (no artificial cap in config)</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">VoxPopuli train subset cap</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>10000</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">LibriSpeech train cap</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>5000</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Batch size</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>16</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">LR SFT / LR RL</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>1e-4</code> / <code>1e-5</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">SFT epochs / RL epochs</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>5</code> / <code>2</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Reward mode / weight / step interval</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>wwer</code> / <code>0.02</code> / every <code>4</code> steps</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Max audio seconds for reward</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>25.0</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Sample rate</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>16000</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">RL objective / grad clip / precision</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>reweight_ctc</code> / <code>1.0</code> / <code>FORCE_FP32: true</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Seed / LoRA / smoke test</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>42</code> / <code>false</code> / <code>false</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Normalize text (eval / SER)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>NORMALIZE_TEXT: false</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Tokenizer UNK guard</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>true</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Debug reward / sample dump</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>false</code> / <code>true</code> (every <code>200</code> steps, <code>10</code> samples)</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Eval toggles</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Zero-shot val, Libri forgetting, final test eval: all <code>true</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Bootstrap iters (paired WER <code>p</code>)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>1000</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Gemini / LLM reward</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>GEMINI_MODEL: gemini-1.5-flash</code>, <code>USE_MOCK_LLM: true</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Domain term weight (WWER)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>3.0</code></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Clinical domain terms</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">37-token list in JSON (e.g. <code>patient</code>, <code>hypertension</code>, <code>malaria</code>, …)</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Parliamentary domain terms</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">13-token list in JSON (e.g. <code>parliament</code>, <code>directive</code>, …)</td></tr>
</tbody>
</table>

**Data loader (training manifests)** — from `nemo/gcp_scripts/nemo_afrispeech_training.py` → `build_data_config`: `max_duration` **20.0** s, `min_duration` **0.5** s, `trim_silence` **false**, `shuffle` **true**, `num_workers` **4** when not smoke test.

---

## 2) AfriSpeech clinical **validation** metrics (`n_utterances` = **1813**)

Transposed so columns stay narrow; values wrap inside cells.

<table style="width:100%;border-collapse:collapse;table-layout:fixed;margin:0.75em 0;">
<thead>
<tr>
<th style="width:22%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Metric</th>
<th style="width:19%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Zero-shot<br/>(base, val)</th>
<th style="width:19%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">After SFT<br/><code>1776207077</code></th>
<th style="width:20%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">After RL<br/><code>1776462369</code></th>
<th style="width:20%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">After RL<br/><code>1776207077</code><br/><small>(collapsed)</small></th>
</tr>
</thead>
<tbody>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">WER (%)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">57.88</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>45.95</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>45.92</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">CER (%)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">25.87</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>14.19</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>14.23</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">SER (%)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">EWER (%)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">19.97</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">20.27</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">18.92</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Domain P / R / F1</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.884 / 0.857 / 0.858</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.881 / 0.867 / 0.861</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.899 / 0.885 / 0.879</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0 / 0 / 0</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Empty hyp frac</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">—</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.0</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Degenerate hyp frac</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">—</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>0.00055</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>0.0</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>1.0</strong></td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Mean hyp len (chars)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">—</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>93.84</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>93.97</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">1.0</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">Train time (s)</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">—</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>12243.1</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>4502.3</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">5007.1</td></tr>
</tbody>
</table>

**RL reward summary (`1776462369`):** `reward_mean` ≈ **0.609**, `reward_std` ≈ **0.076** (batch-level rewards over training; full series in JSON `reward_trajectory`).

---

## 3) Other splits in `1776207077` only (same collapsed RL caveat for RL rows)

<table style="width:100%;border-collapse:collapse;table-layout:fixed;margin:0.75em 0;">
<thead><tr><th style="width:22%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Eval</th><th style="width:22%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Split</th><th style="width:18%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">WER (%)</th><th style="width:18%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">CER (%)</th><th style="width:20%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;"><code>n_utterances</code></th></tr></thead>
<tbody>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">LibriSpeech</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">After SFT</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">10.44</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">4.09</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">2694</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">LibriSpeech</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">After RL</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">2694</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">AfriSpeech clinical</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>test_sft</code></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">50.68</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">17.13</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">3508</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">AfriSpeech clinical</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>test_rl</code></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">100.0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">3508</td></tr>
</tbody>
</table>

<table style="width:100%;border-collapse:collapse;table-layout:fixed;margin:0.75em 0;">
<thead><tr><th style="width:38%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Stat</th><th style="width:62%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Value</th></tr></thead>
<tbody>
<tr>
<td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>paired_bootstrap_pval_sft_vs_rl_wer</code> (<code>1776207077</code>)</td>
<td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><code>0.51</code> — <strong>not meaningful</strong> because RL hypotheses in that run are collapsed.</td>
</tr>
</tbody>
</table>

---

## 4) Training curves (Lightning CSVs)

### SFT (`1776207077` — `*_sft_epoch_metrics.csv`)

End of epoch 4 (train_end): `val_loss` ≈ **30.85**, `val_wer` ≈ **0.255** (NeMo/Lightning **fraction**, not percent).

### RL — healthy run (`1776462369` — `*_rl_epoch_metrics.csv`)

<table style="width:100%;border-collapse:collapse;table-layout:fixed;margin:0.75em 0;">
<thead><tr><th style="width:20%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Epoch</th><th style="width:40%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">train_end <code>val_loss</code></th><th style="width:40%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">train_end <code>val_wer</code> (fraction)</th></tr></thead>
<tbody>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">30.74</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.252</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">1</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">30.81</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.252</td></tr>
</tbody>
</table>

### RL — collapsed run (`1776207077` — `*_rl_epoch_metrics.csv`)

<table style="width:100%;border-collapse:collapse;table-layout:fixed;margin:0.75em 0;">
<thead><tr><th style="width:20%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">Epoch</th><th style="width:40%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">train_end <code>val_loss</code></th><th style="width:40%;border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;background:#f6f8fa;text-align:left;">train_end <code>val_wer</code> (fraction)</th></tr></thead>
<tbody>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>nan</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.939</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">1</td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;"><strong>nan</strong></td><td style="border:1px solid #ccc;padding:8px 10px;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere;">0.939</td></tr>
</tbody>
</table>

---

## 5) How these numbers were computed (brief)

All reported WER/CER/SER/EWER/domain metrics come from `evaluate_manifest_bundle` in `nemo/gcp_scripts/nemo_afrispeech_training.py`:

1. **Transcription:** `model.transcribe(audio_paths, batch_size=CFG.BATCH_SIZE)` with model in **eval** + `torch.no_grad()`.
2. **WER / CER:** `jiwer` via helpers `compute_wer_jiwer` / `compute_cer_jiwer`; stored as **percentage** (`× 100`).
3. **SER:** `sentence_error_rate` — fraction of utterances where **`_normalize_text(ref) != _normalize_text(hyp)`**, with `_normalize_text` = lowercase, strip, collapse whitespace to single spaces. **100% SER** means **no** exact full-string match on that normalization (common under case/punctuation/tokenization drift even when WER is moderate).
4. **EWER:** `entity_wer_from_text` — per utterance, keep only **domain-vocabulary** words that appear in the reference; compute WER on that substring; average over utterances that have ≥1 such reference token; report **mean × 100**. Utterances with no domain tokens in the ref are skipped.
5. **Domain precision / recall / F1:** `aggregate_f1` of per-utterance token sets from `domain_term_precision_recall_f1` (precision uses domain tokens present in hypothesis vs reference domain tokens in ref).
6. **Diagnostics:** `_empty_hyp_frac`, `_degenerate_hyp_frac`, `_mean_hyp_len_chars` — script-side sanity stats on hypothesis strings.

---

## 6) Brief analysis / observations

1. **SFT vs healthy RL (AfriSpeech val):** WER **45.95% → 45.92%** (~0.03 pp), CER **14.19% → 14.23%** (~0.04 pp). This is effectively **flat** on aggregate WER/CER; small gains may appear in **EWER** (20.27% → 18.92%) and domain F1 (0.861 → 0.879), which weight clinical vocabulary more explicitly than plain WER.
2. **SER = 100%** for zero-shot, SFT, and healthy RL is **consistent with strict exact-match SER** under light normalization — it does **not** by itself indicate collapse when WER/CER are healthy.
3. **Run `1776207077` RL is invalid** for reporting (`wer=cer=100`, `_degenerate_hyp_frac=1`, `val_loss=nan` in RL epoch CSV). The **LibriSpeech-after-RL** and **`test_rl`** rows in that JSON reflect the same broken checkpoint.
4. **Run `1776462369` RL is valid** for reporting on AfriSpeech val: finite metrics, **zero** degenerate fraction, normal mean hypothesis length, non-NaN RL epoch metrics in its CSV.
5. **Provenance for the paper:** either (a) re-run a **single** pipeline that writes one `*_results.json` including SFT + **fixed** RL + Libri + test + bootstrap, or (b) keep this split documentation and **do not** mix `1776207077`’s `librispeech_after_rl` / `test_rl` with `1776462369`’s AfriSpeech RL.

---

## 7) On-disk artifacts (local)

- SFT / full pipeline: `vm_results/sft_working_afrispeech_clinical_seed42_1776207077/`
- RL (healthy): `vm_results/afrispeech_clinical_seed42_rl_1776462369/`

Primary JSON paths:

- `vm_results/sft_working_afrispeech_clinical_seed42_1776207077/afrispeech_clinical_seed42_1776207077_results.json`
- `vm_results/afrispeech_clinical_seed42_rl_1776462369/afrispeech_clinical_seed42_rl_1776462369_results.json`
