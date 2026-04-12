# ESPnet2 LibriSpeech Validation Pipeline

Documentation for the practical validation pipeline built around the
LibriSpeech test-clean subset, prediction logging, and reward-model scorer
injection.

Companion files:
- [`egs2/librispeech/asr1/run_librispeech_subset.sh`](../egs2/librispeech/asr1/run_librispeech_subset.sh)
- [`espnet2/bin/asr_inference_with_logging.py`](../espnet2/bin/asr_inference_with_logging.py)
- [`espnet2/asr/scorer/reward_model_scorer.py`](../espnet2/asr/scorer/reward_model_scorer.py)

---

## A. Recipe observations (`run_librispeech_subset.sh`)

### Stage structure of the full `asr.sh` recipe

The LibriSpeech recipe at `egs2/librispeech/asr1/asr.sh` uses a large
`stage` / `stop_stage` guard on every major step:

| Stage | Action |
| --- | --- |
| 1–4 | Data download and preparation (LibriSpeech splits, speed perturbation) |
| 5 | BPE tokenizer training |
| 6–8 | Language model training (optional) |
| 9 | Shape file collection (feature statistics) |
| 10 | ASR training (calls `espnet2.bin.asr_train`) |
| 11 | N-best averaging of model checkpoints |
| **12** | **Decoding** — calls `espnet2.bin.asr_inference` in parallel jobs via `JOB=1:nj` |
| 13 | Score with `score_sclite` (requires SCTK) → `result.wrd.txt` |

The **decoding stage (12)** is what `run_librispeech_subset.sh` replicates
directly, bypassing all training stages by downloading a pretrained model
from `espnet_model_zoo`.

### What the subset script does

1. **Model download** — `espnet_model_zoo` downloads the pretrained Conformer
   (trained on 960h LibriSpeech with speed perturbation) and saves
   `asr_train_config` and `asr_model_file` paths to `exp/asr_conformer_subset/config.txt`.

2. **Subset creation** — `head -n 100 dump/raw/test_clean/wav.scp` produces a
   100-utterance `wav.scp` in `data/test_clean_subset100/`.  The corresponding
   reference transcripts are extracted from `dump/raw/test_clean/text` by
   matching utterance IDs.

3. **Inference** — `espnet2.bin.asr_inference` is called directly in Python
   (no parallel jobs for the subset) with:
   - `beam_size=10` (reduced from the recipe's 60 for speed)
   - `ctc_weight=0.3` (standard Conformer setting)
   - `lm_weight=0.0` (LM skipped; add `--lm_train_config` / `--lm_file` to enable)

4. **Scoring** — `jiwer.wer()` computes both aggregate WER/CER and
   per-utterance WER, written to `exp/asr_conformer_subset/decode_test_clean_subset/result.txt`.

### Expected WER on test-clean

The pretrained Conformer model achieves approximately **2.3–2.5% WER** on
the full test-clean set with beam size 60 and LM rescoring.  Without LM and
with beam size 10, expect **4–6% WER** on the 100-utterance subset (higher
variance due to small sample size).

### Issues and caveats

| Issue | Description | Resolution |
| --- | --- | --- |
| Missing `dump/` directory | If only the model was downloaded (not the full recipe run), `dump/raw/test_clean/wav.scp` does not exist. | Run stages 1–4 of `asr.sh` first, or manually create a `wav.scp` pointing to LibriSpeech FLAC files. |
| SCTK not installed | The standard recipe uses SCTK's `sclite` for scoring. | `run_librispeech_subset.sh` uses `jiwer` as a dependency-free alternative. |
| `espnet_model_zoo` not installed | Required to download the pretrained model. | `pip install espnet_model_zoo` |
| `data/test_clean/text` normalisation | LibriSpeech references are uppercase; the BPE model may produce lowercased output. | Apply `utt2spk`-based text normalisation (`tr '[:upper:]' '[:lower:]'`) or use `jiwer`'s `transforms` to case-fold both sides. |
| GPU not available | The script defaults to `ngpu=0` (CPU). | Set `NGPU=1` and ensure CUDA is available for 5–10× speed-up. |

---

## B. Prediction log structure (`asr_inference_with_logging.py`)

### Where hypotheses are generated

In `asr_inference.py`, the beam search output is produced at:

```664:685:espnet2/bin/asr_inference.py
        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

            last_pos = None if self.asr_model.use_transducer_decoder else -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            token_int = list(filter(lambda x: x != 0, token_int))
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))
```

`asr_inference_with_logging.py` intercepts `results[0]` (1-best hypothesis)
immediately after this call inside the `for keys, batch in loader:` loop and
appends to the JSONL file before moving to the next utterance.

### JSONL record schema

```json
{
  "utt_id":      "1272-128104-0000",
  "hypothesis":  "mister quilter is the apostle of the middle classes",
  "reference":   "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES",
  "wer":         0.0,
  "beam_score":  -14.372,
  "beam_scores": {
    "decoder":       -9.821,
    "ctc":           -4.551,
    "length_bonus":  12.0
  },
  "token":       ["▁MIS", "TER", "▁QUI", "LTER", "▁IS", "▁THE", "..."],
  "token_int":   [1204, 87, 2341, 44, 18, 7, 91, 128, 4, 11, 214, 98]
}
```

| Field | Type | Description |
| --- | --- | --- |
| `utt_id` | string | Utterance ID from the Kaldi wav.scp |
| `hypothesis` | string | 1-best decoded text (after BPE detokenisation) |
| `reference` | string | Ground-truth transcript from `--ref_text_path` |
| `wer` | float | Per-utterance WER via `jiwer.wer(reference, hypothesis)` |
| `beam_score` | float | Total accumulated log-probability of the 1-best hypothesis |
| `beam_scores` | dict | Per-scorer contribution breakdown (decoder, ctc, lm, …) |
| `token` | list[str] | BPE token strings (sub-word level) |
| `token_int` | list[int] | Integer token IDs |

The `wer` field is omitted when `jiwer` is not installed or when
`--ref_text_path` is not provided.  The `reference` field is an empty string
in that case.

### Reward simulation from the log

The JSONL log supports reward computation simulation:
```python
import json, jiwer

with open("predictions.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        reward = max(0.0, 1.0 - entry["wer"])   # REINFORCE reward
        log_prob = entry["beam_score"]           # log π(y|x) (unnormalised)
        pg_loss_contribution = -reward * log_prob
```

This is exactly the quantity computed online in `RLESPnetModel._compute_pg_loss()`,
making the log useful for offline reward distribution analysis before committing
to a full RL training run.

### Usage

```bash
python -m espnet2.bin.asr_inference_with_logging \
    --ngpu 0 \
    --beam_size 10 \
    --ctc_weight 0.3 \
    --data_path_and_name_and_type \
        data/test_clean_subset100/wav.scp,speech,sound \
    --asr_train_config exp/asr_conformer_subset/config.yaml \
    --asr_model_file   exp/asr_conformer_subset/valid.acc.best.pth \
    --output_dir       exp/decode/test_clean_subset \
    --ref_text_path    data/test_clean_subset100/text \
    --prediction_log_path exp/decode/test_clean_subset/predictions.jsonl
```

---

## C. Reward scorer integration with beam search

### Scorer registration in `BeamSearch`

`BeamSearch.__init__()` (line 82–95 of `beam_search.py`) registers all
scorers at construction time:

```python
for k, v in scorers.items():
    w = weights.get(k, 0)
    if w == 0 or v is None:
        continue
    assert isinstance(v, ScorerInterface)
    self.scorers[k] = v
    if isinstance(v, PartialScorerInterface):
        self.part_scorers[k] = v
    else:
        self.full_scorers[k] = v
    if isinstance(v, torch.nn.Module):
        self.nn_dict[k] = v
```

A scorer is a **full scorer** if it implements `ScorerInterface` but not
`PartialScorerInterface`.  It is a **partial scorer** (called only on
pre-beam tokens) if it implements `PartialScorerInterface`.

`RewardModelScorer` implements `ScorerInterface` directly → it becomes a
**full scorer** and is called on every vocabulary token at every beam step.

### Per-step scoring mechanics

Inside `BeamSearch.search()` (line 316–381), each hypothesis is scored:

```python
for hyp in running_hyps:
    weighted_scores = torch.zeros(n_vocab, ...)
    scores, states = self.score_full(hyp, x)          # all full scorers
    for k in self.full_scorers:
        weighted_scores += self.weights[k] * scores[k]  # weighted sum
    # ... partial scorers add on top ...
    weighted_scores += hyp.score                      # cumulative score
    # top-beam tokens selected from weighted_scores
```

`RewardModelScorer.score()` contributes a **uniform bonus** across all
vocabulary positions.  This means:
- It does not change the relative ranking of next tokens at a given step
  (all get the same bonus).
- It **does** shift the cumulative beam score, influencing which hypotheses
  survive pruning across steps.
- Over multiple steps, hypotheses with consistently higher reward accumulate
  a larger total bonus, allowing them to survive pruning even if their
  language-model or CTC scores are slightly lower.

This is the correct semantics for hypothesis-level reward models: the reward
evaluates the prefix as a whole, not the individual next token.

### Scorer hierarchy

```
ScorerInterface               (base — required: score())
    ├── BatchScorerInterface  (optional: batch_score() for GPU parallelism)
    │       └── LengthBonus, CTCPrefixScorer (batch-capable)
    ├── PartialScorerInterface (called only on pre-beam tokens)
    │       └── CTCPrefixScorer
    └── RewardModelScorer     (full scorer, no batching — adds reward bonus)
```

### Injecting the scorer without modifying `asr_inference.py`

The cleanest approach is to subclass `Speech2Text`:

```python
from espnet2.bin.asr_inference import Speech2Text
from espnet2.asr.scorer.reward_model_scorer import (
    RewardModelScorer, mock_no_oov_reward_fn
)

class Speech2TextWithReward(Speech2Text):
    def __init__(self, *args, reward_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if self.beam_search is not None:
            scorer = RewardModelScorer(
                reward_fn=mock_no_oov_reward_fn,
                token_list=self.asr_model.token_list,
                tokenizer=self.tokenizer,
                vocab_size=len(self.asr_model.token_list),
                blank_id=self.asr_model.blank_id,
                sos_id=self.asr_model.sos,
                eos_id=self.asr_model.eos,
            )
            self.beam_search.scorers["reward_model"] = scorer
            self.beam_search.full_scorers["reward_model"] = scorer
            self.beam_search.weights["reward_model"] = reward_weight
```

### YAML config snippet

```yaml
# conf/decode_asr_with_reward.yaml
beam_size: 10
ctc_weight: 0.3
lm_weight: 0.0
penalty: 0.0      # weight for length_bonus

# Custom scorer weights are passed via the 'weights' dict in BeamSearch,
# not directly in the decode config YAML (no native YAML key for custom scorers).
# Use the Speech2TextWithReward subclass above and pass reward_weight=0.1.
#
# Conceptual mapping:
# beam_search_conf:
#   scorers:
#     decoder: full      # TransformerDecoder — weight = 1 - ctc_weight
#     ctc: 0.3           # CTCPrefixScorer    — weight = ctc_weight
#     reward_model: 0.1  # RewardModelScorer  — weight set in code
```

---

## D. Issues encountered and recommendations

### Issue 1: `BatchBeamSearch` falls back when `RewardModelScorer` is added

`asr_inference.py` lines 376–396 check whether all full scorers implement
`BatchScorerInterface`.  `RewardModelScorer` only implements `ScorerInterface`,
so the batch upgrade is blocked and the slower iterative `BeamSearch` is used.

**Resolution:** Implement `batch_score()` in `RewardModelScorer` to enable
`BatchBeamSearch`:

```python
def batch_score(self, ys, states, xs):
    bonus = sum(float(self.reward_fn(self._detokenize(y))) for y in ys) / len(ys)
    scores = torch.full((ys.shape[0], self.vocab_size), bonus, ...)
    return scores, [None] * len(ys)
```

For a stateless uniform-bonus scorer, each row can be computed independently
so the batch implementation is trivial and cheap.

### Issue 2: Reward function latency

The reward function is called **once per hypothesis per beam step** (i.e.
`beam_size × sequence_length` times per utterance).  Slow reward functions
(e.g. neural LM perplexity requiring a GPU forward pass) can make decoding
prohibitively slow.

**Mitigation strategies:**
- Cache the reward by prefix hash: reward only changes when a new token is
  added, but the prefix may be shared across beam hypotheses.
- Compute the reward only at EOS using `final_score()` rather than at every
  step.  This is an approximation but avoids per-step overhead.
- Use a small distilled reward model (e.g. 2-layer MLP on top of CTC features)
  that runs in < 1 ms per call.

### Issue 3: Reward scale mismatch

Log-probabilities from the decoder and CTC are typically in the range
`[-50, 0]` per utterance, while the mock reward returns values in `[0, 0.1]`.
A weight of `0.1` on the reward scorer contributes `0.01` per step, which is
negligible compared to decoder contributions of `~5` per step.

**Resolution:** Normalise the reward to the expected score range before use,
or increase `reward_weight` significantly (e.g. `5.0–20.0`) when using
small-scale reward signals.

### Issue 4: Text normalisation mismatch between hypothesis and reference

LibriSpeech references are uppercase; the ESPnet2 BPE model may produce
mixed-case output.  `jiwer.wer()` is case-sensitive by default, inflating WER.

**Resolution:** Apply `jiwer.transforms.ToLowerCase()` + `RemovePunctuation()`
to both reference and hypothesis before calling `jiwer.wer()`:

```python
import jiwer
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])
wer = jiwer.wer(ref, hyp, truth_transform=transform, hypothesis_transform=transform)
```

### Issue 5: `espnet_model_zoo` downloads to `~/.cache`

The pretrained model is stored in `~/.cache/espnet_model_zoo/` and can be
several gigabytes.  Use `ModelDownloader(cachedir="/path/to/cache")` to
redirect the download location.
