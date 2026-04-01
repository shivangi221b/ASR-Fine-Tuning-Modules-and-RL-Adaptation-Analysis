# ESPnet2 RL Compatibility Assessment

Assessment of ESPnet2's architecture for reinforcement learning extensions,
covering reward injection feasibility, multi-task loss blending, distributed
training compatibility, and recommended RL algorithm.

Companion files:
- [`espnet2/train/rl_trainer.py`](../espnet2/train/rl_trainer.py) — `RLTrainer` extending the base trainer
- [`espnet2/asr/rl_espnet_model.py`](../espnet2/asr/rl_espnet_model.py) — `RLESPnetModel` with RL forward pass

---

## 1. `train_one_epoch()` lifecycle

### Full step sequence (from source)

```
for each mini-batch:
  ① data loading         iterator yields (utt_ids, batch_dict)
  ② device transfer      to_device(batch, cuda)
  ③ forward (autocast)   retval = model(**batch)
  ④ stats registration   reporter.register(stats, weight)
  ⑤ backward             loss.backward()  [or scaler.scale(loss).backward()]
  ⑥ [every accum_grad steps]:
       scaler.unscale_()   (AMP only)
       grad_noise          (optional)
       clip_grad_norm_     global clip over all parameters
       optimizer.step()
       scheduler.step()    (batch-step schedulers only)
       optimizer.zero_grad()
       reporter.register(lr, train_time)
  ⑦ reporter.next()

for each epoch boundary:
  ⑧ epoch-step schedulers: scheduler.step() / scheduler.step(val_metric)
  ⑨ checkpoint save
  ⑩ early-stopping check
```

### RL reward injection point

The reward must be injected **between steps ③ and ⑤** — after the forward
pass produces model outputs and before the backward pass computes gradients.

In ESPnet2, step ③ is `retval = model(**batch)`.  The model is the only
entity that sees both the speech input and the reference text, making it the
natural site for:
- greedy CTC decoding (rollout)
- WER computation (reward)
- policy-gradient loss computation

The injection is implemented by passing `rl_weight` through the batch dict:
```python
batch["rl_weight"] = rl_weight      # injected by RLTrainer before model call
retval = model(**batch)             # RLESPnetModel reads rl_weight from **kwargs
```

This keeps DDP's all-reduce intact because there is a single DDP-wrapped
forward call per step.

### `stats` dict aggregation and logging

The model populates a `stats: Dict[str, Optional[Tensor]]` dict, which is
returned alongside `loss` and `weight`.  The `Trainer` then:
1. Passes it to `reporter.register(stats, weight)` after each mini-batch.
2. The `SubReporter` accumulates weighted averages across the batch and across
   `accum_grad` micro-batches using its internal `WeightedAverage` objects.
3. At log intervals, `reporter.log_message()` formats a string of all tracked
   scalars and logs it.
4. If TensorBoard or W&B are enabled, scalars are forwarded automatically.

Any key added to `stats` (e.g. `reward`, `pg_loss`, `ce_loss`) is tracked
automatically without any changes to the reporter or logger.

### Gradient clipping

`torch.nn.utils.clip_grad_norm_` is called **globally** across all model
parameters at lines 732–736 of `trainer.py`:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=grad_clip,
    norm_type=grad_clip_type,
)
```

This applies to the blended gradient (CE + PG combined), which is correct
for REINFORCE since both losses produce gradients over the same parameters.
There is no per-branch clipping — if separate clipping is needed, override
the section between the `unscale_` call and `optimizer.step()`.

---

## 2. `AbsESPnetModel.forward()` contract

### Signature

```python
@abstractmethod
def forward(self, **batch: torch.Tensor) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
    ...
```

### Required return values

| Position | Type | Semantics |
| --- | --- | --- |
| 0 | `Tensor` (scalar) | Loss to backward. Must be differentiable. |
| 1 | `Dict[str, Optional[Tensor]]` | Per-scalar stats for logging; values are **detached** before logging. |
| 2 | `Tensor` (scalar or 1-d) | Batch weight for weighted averaging across DataParallel replicas. Typically `batch_size`. |

Alternatively the model can return a `dict` with keys `"loss"`, `"stats"`,
`"weight"`, and optionally `"optim_idx"` (int) to route the gradient to a
specific optimizer.

### Hooks and callbacks

There are **no pre/post-forward hooks** defined in `AbsESPnetModel` itself.
It inherits from `torch.nn.Module`, so standard PyTorch hooks
(`register_forward_hook`, `register_backward_hook`) apply.

`force_gatherable((loss, stats, weight), device)` — called at the end of
every concrete `forward()` — is the only fixed post-processing step.  It
converts Python scalars/floats to 1-d tensors and moves everything to the
specified device for DataParallel gathering.

---

## 3. Multi-task loss blending — CTC + attention template

`ESPnetASRModel.forward()` (lines 362–367 of `espnet_model.py`) blends two
supervised losses via a scalar coefficient:

```python
if self.ctc_weight == 0.0:
    loss = loss_att
elif self.ctc_weight == 1.0:
    loss = loss_ctc
else:
    loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
```

The same two-coefficient pattern applies identically to CE + RL blending:

```python
total_loss = (1.0 - rl_weight) * ce_loss + rl_weight * pg_loss
```

Key properties of this pattern:
- Both terms must be scalar tensors on the same device.
- Both must be differentiable w.r.t. model parameters (gradients flow through
  both terms simultaneously in a single `backward()`).
- `rl_weight` can be annealed during training (e.g. warm-up the RL signal
  after initial CE convergence) by changing the value passed in the batch dict.
- A three-way blend (CTC + attention + RL) is also possible:
  ```
  loss = w_ctc * loss_ctc + w_att * loss_att + w_rl * pg_loss
  ```
  subject to `w_ctc + w_att + w_rl = 1`.

Each loss component is stored in `stats` with `.detach()` so that the
`Reporter` does not retain computation graphs.

---

## 4. Distributed training compatibility

### Current DDP setup

`Trainer.run()` wraps the model in `DistributedDataParallel`:
```python
dp_model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[torch.cuda.current_device()],   # 1-GPU-per-process mode
    find_unused_parameters=trainer_options.unused_parameters,
    gradient_as_bucket_view=trainer_options.gradient_as_bucket_view,
)
```

`DistributedOption` supports:
- **NCCL backend** (default) over TCP or env-based rendezvous.
- **SLURM** launcher (reads `SLURM_PROCID`, `SLURM_NTASKS`, etc.).
- **MPI** launcher via `mpi4py`.
- **env://** launch (PyTorch-native `torchrun` / `torch.distributed.launch`).
- **DeepSpeed** via `init_deepspeed()` (requires prior torch distributed init).

Stats are aggregated across ranks with `recursive_average()` which calls
`torch.distributed.all_reduce` internally.

### RL-specific distributed concerns

#### A. Reward computation is per-rank, not globally reduced

Each rank decodes its own shard of the batch independently using
`jiwer.wer()` (CPU, no network I/O).  The resulting per-utterance rewards
feed into `pg_loss`, which is reduced by `all_reduce` as part of normal DDP
gradient synchronisation.  No extra collective operations are needed.

#### B. `forward_rl()` must NOT be called through DDP

DDP intercepts `model(...)` calls (i.e. `__call__` → `forward`) and inserts
gradient hooks for all-reduce.  Calling `model.module.forward_rl(...)` bypasses
DDP, so gradients from that call are **not** all-reduced across ranks.

For the `RLESPnetModel` design used in `RLTrainer`, this is not an issue
because the RL computation happens inside `forward()` (the single DDP-wrapped
call).  `forward_rl()` is provided for PPO-style offline rollouts where
gradient synchronisation is handled explicitly.

#### C. Multi-node rollout collection for PPO

For PPO or IMPALA, where trajectories are collected asynchronously across
actors and a separate learner updates the model, a custom rollout aggregation
step (e.g. `dist.gather` of `seq_log_probs`, `rewards`) is required.  This
goes beyond the scaffold and would require:
1. A dedicated `rollout_one_epoch()` method (no gradient, no optimizer step).
2. A `learn_one_epoch()` method that consumes stored rollouts with PPO's
   clipped surrogate objective.
3. Careful reference-model management (old policy π_old for importance weights).

None of these require changes to `DistributedOption`; they use the same NCCL
process group.

#### D. `find_unused_parameters` consideration

If `rl_weight == 0` for some steps and `> 0` for others (e.g. scheduled RL
warmup), some RL-specific parameters may not receive gradients in the CE-only
steps.  Set `find_unused_parameters: true` in your config to avoid DDP errors.

---

## 5. Recommended RL algorithm: REINFORCE vs PPO

### REINFORCE (Monte Carlo policy gradient)

**Compatibility: High**

REINFORCE requires only:
1. A single forward pass with teacher-forcing disabled (greedy or sampled decode).
2. A scalar reward per utterance.
3. `pg_loss = -reward * log_prob_of_hypothesis`.

It fits directly into ESPnet2's existing training loop because:
- The PG loss is computed inside `model.forward()` — no change to the trainer loop
  beyond injecting `rl_weight`.
- Gradient flow is identical to the CE case: one backward per mini-batch.
- DDP gradient synchronisation requires zero extra code.
- The `stats` dict extension (`reward`, `pg_loss`, `ce_loss`) is a one-liner.

**Limitations:**
- High variance: the REINFORCE gradient estimator has variance `O(T)` in the
  sequence length `T`.  Mitigated by a WER reward baseline (subtract the mean
  reward across the batch) or a self-critical baseline (`reward - greedy_reward`).
- Greedy decoding is biased: the "policy" is deterministic so there is no
  exploration.  Stochastic sampling (temperature > 1) improves exploration but
  increases sequence length variance.
- Reward is discrete (WER is piece-wise constant): small parameter changes
  that don't change the WER still receive a zero gradient signal, slowing
  convergence.

**Practical recommendation for REINFORCE:**
```
rl_weight: 0.1 (start small, anneal up)
Baseline: subtract batch-mean reward from each per-utterance reward
Reward: 1 - WER (using jiwer)
Warmup: CE-only for first N steps, then ramp rl_weight linearly
```

### PPO (Proximal Policy Optimisation)

**Compatibility: Medium — requires scaffolding extensions**

PPO offers significantly lower variance than REINFORCE because of:
1. Multiple gradient steps per rollout (data efficiency).
2. Clipped surrogate objective preventing destructive large updates.
3. Separate value network (critic) for advantage estimation.

However, integrating PPO into ESPnet2 requires:

| Component | ESPnet2 support | Work needed |
| --- | --- | --- |
| Rollout collection | Partially (via `forward_rl()`) | Dedicated `rollout_one_epoch()`, buffer |
| Reference (old) policy | Not present | Snapshot of model weights; importance weight `π/π_old` |
| Value network (critic) | Not present | New MLP head on `encoder_out`; separate optimizer |
| PPO objective | Not present | Clipped surrogate + entropy bonus |
| Rollout buffer | Not present | In-memory buffer for (state, action, log_prob, reward, advantage) |
| GAE (advantage estimation) | Not present | Generalised Advantage Estimation loop |

PPO would require a new `PPOTrainer` that separates rollout collection from
learning steps, and a new `PPOASRModel` that adds a value head.  The
`DistributedOption` infrastructure is reusable as-is.

**When to choose PPO:**
- You have a strong language model prior and want to optimise a compound metric
  (e.g. BLEU + WER + speaker similarity) over many episodes per utterance.
- Your rewards are slow to compute (e.g. downstream NLU evaluation) and you
  want to amortise the cost over multiple gradient steps.
- Training corpus is small and sample efficiency is critical.

**When to stay with REINFORCE:**
- You are first validating that the RL signal actually helps WER.
- You want minimal engineering risk and maximum compatibility with existing
  ESPnet2 tooling (recipes, multi-GPU setup, reporter, checkpoint saving).
- Your CE-pretrained model is already strong; you only need a small RL push.

### Minimum viable RL recipe (REINFORCE)

1. Pretrain with standard CE loss to convergence.
2. Resume with `RLTrainer` and `RLESPnetModel`:
   ```yaml
   model: rl_espnet
   trainer: rl_trainer
   rl_weight: 0.05    # injected into batch dict; start small
   max_epoch: 10      # RL fine-tuning epochs
   optim_conf:
     lr: 1.0e-5       # lower LR for RL phase
   scheduler: none    # constant LR preferred for RL stability
   ```
3. Monitor `reward` and `pg_loss` in TensorBoard alongside `loss_att` and
   `loss_ctc`; expect reward to climb slowly over the first 2–3 epochs.
4. If reward plateaus, add a self-critical baseline:
   ```python
   # In _compute_pg_loss: subtract greedy baseline reward
   baseline_reward = rewards.mean().detach()
   pg_loss = -((rewards.detach() - baseline_reward) * seq_log_probs).mean()
   ```

---

## Summary

| Topic | Finding |
| --- | --- |
| **Reward injection** | Clean: inject `rl_weight` in batch dict; `RLESPnetModel.forward()` handles both CE and PG in one DDP call. |
| **Loss blending template** | Identical to CTC+attention: `total = (1-w)*ce + w*pg`. No trainer changes beyond batch augmentation. |
| **Stats logging** | Transparent: any key added to `stats` dict is automatically tracked by ESPnet2's Reporter. |
| **Gradient clipping** | Applied globally after blended backward — correct for REINFORCE; does not need modification. |
| **DDP compatibility** | Full: single forward call preserves all-reduce. `forward_rl()` explicitly noted as bypass-of-DDP for PPO. |
| **Multi-node** | Reuse existing `DistributedOption` + NCCL; no extra collectives needed for REINFORCE. |
| **Recommended algorithm** | **REINFORCE** for near-term integration; **PPO** for long-term if sample efficiency becomes critical. |
| **Key risk** | REINFORCE high variance — mitigate with self-critical baseline and small `rl_weight`. |
