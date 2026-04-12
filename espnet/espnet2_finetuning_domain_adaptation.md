# ESPnet2: Fine-Tuning and Domain Adaptation Notes

This document summarizes how ESPnet2 handles training loops, learning-rate schedulers, recipe configs, and ASR model construction for fine-tuning workflows. A companion YAML template lives at [`train_asr_conformer_finetune_template.yaml`](train_asr_conformer_finetune_template.yaml).

---

## 1. `train_one_epoch()` in `espnet2/train/trainer.py`

### Gradient accumulation and optimizer / scheduler sequencing

The core training loop accumulates `accum_grad` mini-batches before taking a parameter update. The sequence for each mini-batch step `iiter` is:

1. **Forward pass** (inside `autocast` for AMP): `loss = model(**batch)`.
2. **Scale by `accum_grad`**: `loss /= accum_grad` — each micro-batch contributes `1/accum_grad` of the true gradient.
3. **Backward pass**: `scaler.scale(loss).backward()` (AMP) or `loss.backward()`.
4. **Every `accum_grad` steps** — the update gate opens:
   - `scaler.unscale_(optimizer)` (AMP only).
   - Optional gradient noise (`grad_noise`).
   - `clip_grad_norm_` over **all** model parameters to `grad_clip`.
   - If `grad_norm` is non-finite, the step is skipped.
   - **Optimizer step**: `scaler.step(optimizer)` / `optimizer.step()`.
   - **Batch-step scheduler**: `scheduler.step()` only for `AbsBatchStepScheduler`, immediately after the optimizer step.
   - `optimizer.zero_grad()`.
5. **After each full epoch**, in `run()`, epoch-level schedulers (`AbsEpochStepScheduler` / `AbsValEpochStepScheduler`) call `.step()` once.

**Implication:** `clip_grad_norm_` is global; there is no per-parameter-group clipping in the default trainer.

Relevant excerpt:

```772:787:espnet2/train/trainer.py
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
```

### Multi-optimizer support

`optimizers` and `schedulers` are lists. The model can return `optim_idx` to route a batch to a specific optimizer. The reporter logs `optim{i}_lr{j}` for every parameter group.

### Warmup scheduler placement

Warmup schedulers (`WarmupLR`, `NoamLR`, etc.) subclass `AbsBatchStepScheduler` and step **once per effective optimizer step** (every `accum_grad` mini-batches), not per raw mini-batch.

### Per-layer / per-parameter-group learning rates

`ASRTask` does not build per-layer parameter groups by default. The trainer supports PyTorch `param_groups` (multiple LRs are logged). Differential LRs require a custom task or overriding `build_optimizer()` in `abs_task.py`.

---

## 2. LR schedulers in `espnet2/schedulers/`

Custom schedulers are batch-step (`AbsBatchStepScheduler`) unless noted.

| Class | Parameters | Behaviour |
| --- | --- | --- |
| **WarmupLR** | `warmup_steps` (default 25000) | Ramps to `base_lr`, then inverse-square-root decay. Recommended default. |
| **NoamLR** (deprecated) | `model_size` (320), `warmup_steps` (25000) | Original Transformer schedule; use WarmupLR + adjusted `base_lr` instead. |
| **TristageLR** | `max_steps`, `warmup_ratio`, `hold_ratio`, `decay_ratio`, `init_lr_scale`, `final_lr_scale` | Linear warmup → hold → exponential decay. |
| **WarmupStepLR** | `warmup_steps`, `steps_per_epoch`, `step_size`, `gamma` | Warmup then StepLR-style decay by epoch. |
| **CosineAnnealingWarmupRestarts** | `first_cycle_steps`, `cycle_mult`, `max_lr`, `min_lr`, `warmup_steps`, `gamma` | Cosine cycles with warmup. |
| **PiecewiseLinearWarmupLR** | `warmup_steps_list`, `warmup_lr_list` | Piecewise linear warmup then decay. |
| **ExponentialDecayWarmup** | `max_lr`, `min_lr`, `total_steps`, `warmup_steps`, `warm_from_zero` | Optional linear warmup from zero, then exponential decay. |
| **WarmupReduceLROnPlateau** | `warmup_steps`, `mode`, `factor`, `patience`, `threshold`, `cooldown`, `min_lr`, `eps` | Warmup then validation-metric plateau reduction. |

PyTorch built-ins are also registered: epoch-step (`ReduceLROnPlateau`, `StepLR`, …) and batch-step (`OneCycleLR`, `CosineAnnealingWarmRestarts`, …). See `espnet2/schedulers/abs_scheduler.py`.

---

## 3. `train_*.yaml` in LibriSpeech and AiShell

### LibriSpeech (`egs2/librispeech/asr1/conf/`)

| Config | Encoder | Decoder | `ctc_weight` | Optimizer | LR | Scheduler | `warmup_steps` | SpecAug |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train_asr_conformer.yaml` | conformer (12×512) | transformer (6) | 0.3 | adam | 0.0025 | warmuplr | 40000 | Yes |
| `train_asr_transformer.yaml` | transformer (12×512) | transformer (6) | 0.3 | adam | 0.002 | warmuplr | 25000 | Yes |
| `train_asr_branchformer.yaml` | branchformer (18×512) | transformer (6) | 0.3 | adam | 0.0025 | warmuplr | 40000 | Yes |
| `train_asr_rnnt.yaml` | conformer (12×512) | transducer | 0.3 | adam | 0.0015 | warmuplr | 25000 | Yes |

LM configs (`train_lm_transformer.yaml`, `train_rnn_lm.yaml`) are separate from ASR and were not expanded in the original survey.

### AiShell (`egs2/aishell/asr1/conf/`)

| Config | Encoder | Decoder | `ctc_weight` | Optimizer | LR | Scheduler | `warmup_steps` | SpecAug |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train_asr_conformer.yaml` | conformer (12×256) | transformer (6) | 0.3 | adam | 0.001 | warmuplr | 35000 | Yes |
| `train_asr_transformer.yaml` | transformer (12×256) | transformer (6) | 0.3 | adam | 0.002 | warmuplr | 25000 | No |
| `train_asr_branchformer.yaml` | branchformer (24×256) | transformer (6) | 0.3 | adam | 0.001 | warmuplr | 35000 | Yes |
| `train_asr_streaming_conformer.yaml` | contextual_block_conformer | transformer (6) | 0.3 | adam | 0.0005 | warmuplr | 30000 | Yes |
| `train_asr_rnn.yaml` | vgg_rnn | rnn | 0.5 | adadelta | 1.0 | reducelronplateau | n/a | No |

None of these reference configs use `init_param` or `freeze_param`; those are typically passed on the command line or added to YAML where supported.

---

## 4. `ASRTask.build_model()` in `espnet2/tasks/asr.py`

### Pipeline

1. Resolve `token_list` and `vocab_size`.
2. **Frontend** (unless `input_size` is set): `frontend_class(**args.frontend_conf)` → `input_size`.
3. **SpecAug** (optional): `SpecAug(**args.specaug_conf)`.
4. **Normalize** (optional).
5. **Preencoder** (optional).
6. **Encoder**: `encoder_class(input_size=input_size, **args.encoder_conf)`.
7. **Postencoder** (optional).
8. **Decoder**: transducer gets special args; otherwise `decoder_class(vocab_size=..., encoder_output_size=..., **args.decoder_conf)`.
9. **CTC**: `CTC(odim=vocab_size, encoder_output_size=..., **args.ctc_conf)`.
10. **Model**: `model_class(..., **args.model_conf)`.
11. **Init** (optional): `initialize(model, args.init)`.

### YAML → constructor mapping

- `encoder: <name>` selects a class from `encoder_choices`; `encoder_conf` keys are passed as kwargs (plus `input_size` for the encoder).
- `decoder: <name>` selects from `decoder_choices`; `decoder_conf` is merged with required ctor args (`vocab_size`, `encoder_output_size`, etc.).
- `model_conf` is passed into `ESPnetASRModel` (or chosen model class), e.g. `ctc_weight`, `lsm_weight`.

### Registered encoders (short list)

`conformer`, `transformer`, `branchformer`, `e_branchformer`, `contextual_block_transformer`, `contextual_block_conformer`, `vgg_rnn`, `rnn`, `wav2vec2`, `hubert`, `whisper`, `longformer`, `multiconv_conformer`, `beats`, and related HuBERT / multispkr variants — see `encoder_choices` in `asr.py`.

### Registered decoders

`transformer`, `rnn`, `transducer`, convolution variants, `mlm`, `whisper`, `hugging_face_transformers`, `s4`, `linear_decoder`, etc.

### `init_param` / `freeze_param` (in `abs_task.py`)

- **`init_param`**: load pretrained weights; format `<path>[:src_key:dst_key:exclude_keys]`.
- **`freeze_param`**: list of name prefixes; any `named_parameters()` matching `t` or `t.` gets `requires_grad = False`.

For a 12-layer Conformer, bottom six blocks are `encoder.encoders.0` … `encoder.encoders.5`.

---

## 5. Fine-tuning YAML template

The full template is in **[`train_asr_conformer_finetune_template.yaml`](train_asr_conformer_finetune_template.yaml)** in this directory. Adjust `init_param` path, `encoder_conf` / `decoder_conf` to match the checkpoint, and training hyperparameters for your data.

---

## Summary table

| Topic | Finding |
| --- | --- |
| Gradient accumulation | `loss /= accum_grad`; optimizer and batch schedulers step every `accum_grad` iterations. |
| Warmup steps | Count **effective** updates, not every mini-batch. |
| Epoch vs batch schedulers | Batch schedulers step inside the training loop; epoch schedulers after validation. |
| Per-layer LRs | Not default in ASR task; use custom `param_groups`. |
| Pretrained load | `--init_param` with optional key remapping. |
| Freezing | `--freeze_param` prefix match on parameter names. |
| SpecAugment | `specaug: specaug` + `specaug_conf` → `SpecAug(**specaug_conf)`. |
| Preferred warmup scheduler | **WarmupLR**; **NoamLR** is deprecated in-tree. |
