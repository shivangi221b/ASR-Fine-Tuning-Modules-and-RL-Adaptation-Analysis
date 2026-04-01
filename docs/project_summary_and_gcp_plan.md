# STT Domain Adaptation with RL — Complete Project Summary
## For GCP / Local Execution

---

## PART 1: WHAT THIS PAPER IS ABOUT

### The Core Claim
Standard supervised fine-tuning (SFT) is the dominant method for adapting 
speech-to-text (STT) models to specialised domains. This paper argues that 
reinforcement learning (RL) applied ON TOP of SFT — using domain-aware reward 
signals — improves transcription of domain-specific terminology beyond what 
SFT alone achieves. We test this across two frameworks, two domain datasets, 
and three reward designs.

### The Research Gaps We Fill
Nobody has published:
1. A systematic comparison of RL reward methods (MWER, WWER, LLM-based) 
   on healthcare AND judicial domain audio
2. A framework comparison (HuggingFace Transformers vs NVIDIA NeMo) for 
   domain-specific STT with RL
3. Reward design principles for domain-specific STT

These are all explicitly unpublished as of April 2026. Any one of them is 
publishable. All three together is a strong paper.

### What We Are NOT Doing
- NOT using VERL, MILES, AReaL — these are LLM-only RL frameworks, 
  architecturally incompatible with audio encoding
- NOT running RL from scratch — always SFT first, then RL on top
- NOT claiming RL is a silver bullet — expected gains are 2-5% WER 
  improvement, with larger gains on domain term accuracy (EWER)

---

## PART 2: PAPER REQUIREMENTS CHECKLIST

### 2.1 Models

#### HuggingFace Framework
- [ ] **Primary:** `openai/whisper-small` (244M params)
  - Encoder-decoder architecture
  - Better baseline WER than whisper-base, more headroom for RL improvement
  - Fits on A100 40GB with fp16
- [ ] **SOTA upper bound (optional):** `openai/whisper-large-v3` (1.5B params)
  - Only if time/budget allows on GCP
  - Shows method generalises to SOTA

#### NeMo Framework  
- [ ] **Primary:** `stt_en_conformer_ctc_medium` (30M params)
  - CTC architecture — different from Whisper's encoder-decoder
  - Comparable scale to whisper-small for fair framework comparison
  - Hook point: `_compute_loss()` override
- [ ] **SOTA upper bound (optional):** `nvidia/parakeet-ctc-1.1b`
  - Drop-in replacement in NeMo (one line change)
  - Only if time/budget allows

#### LoRA / PEFT
- [ ] LoRA should be tested as a parameter-efficient alternative to full fine-tuning
  - HF: use `peft` library, target Whisper's attention layers, rank=32
  - NeMo: model transform callback, target encoder linear layers, rank=32
  - Compare: full fine-tuning WER vs LoRA WER
  - Expected: LoRA ~1-2% worse WER but 98% fewer trainable params
  - This is a secondary finding — document it but don't let it block primary results
  - For GCP: run full fine-tuning first, LoRA as a second experiment

### 2.2 Datasets

#### Domain-Specific (Primary — these are your paper's main contribution)
- [ ] **Healthcare:** `tobiolatunji/afrispeech-200` (clinical subset)
  - African-accented English medical speech
  - Has medical entity annotations (medications, anatomy, conditions, procedures)
  - Supports M-WER and M-CER metrics
  - text field: `transcript`, filter: `domain == "clinical"`
  - Load: `load_dataset("tobiolatunji/afrispeech-200", "all")` (or `intronhealth/afrispeech-200`)
  - **Publication protocol:** official `train` / `validation` / `test` splits; filter `domain == "clinical"` on each. Use **all** clinical training clips (~36k) unless ablating size.
  - Script default: full clinical train/val/test (caps optional via `--train_samples` etc.)
  
- [ ] **Parliamentary / formal-speech (proxy):** `facebook/voxpopuli` (English subset)
  - European Parliament proceedings — formal register, policy terminology (not courtroom speech)
  - No login required, CC0 licence
  - text field: `normalized_text` (script)
  - Target: **10,000 train** sampled with fixed `--seed` (not first-N chronological); official `validation` split for dev metrics

- [ ] **Healthcare augmentation (optional):** `united-we-care/United-Syn-Med`
  - Synthetic TTS-generated medical speech — must disclose in paper
  - Use to augment training data if AfriSpeech clinical subset is too small
  - text field: `text`

#### General English Baseline (Required for comparison)
- [ ] `librispeech_asr` clean-100
  - Catastrophic-forgetting check on `validation` (script builds eval manifest)
  - text field: `text`
  - Script default: up to 5k train for Libri-specific SFT if used; full val slice for forgetting eval

### 2.3 Training Pipeline

#### Stage 1: SFT (Supervised Fine-Tuning)
- [ ] Pure cross-entropy / CTC loss — no reward signal
- [ ] Train to convergence (not just 1-2 epochs)
  - HF: 5 epochs on full clinical train (or same protocol as NeMo)
  - NeMo: 5 epochs (default in `nemo_afrispeech_training.py`)
- [ ] Save best checkpoint based on validation WER
- [ ] Record: train loss per epoch, val loss per epoch, WER per epoch
- [ ] This is your baseline — every RL result is compared against this

#### Stage 2: RL Fine-tuning (on top of SFT checkpoint)
Run three separate RL experiments, each starting from the SAME SFT checkpoint:

- [ ] **RL-MWER:** reward = 1 - WER, weight = 0.05, 2 epochs, LR = 1/10th of SFT LR
- [ ] **RL-WWER:** same as MWER but domain terms weighted 3x, 2 epochs
- [ ] **RL-LLM:** mock LLM scorer in code by default; real scorer via **Google Gemini API** (`GEMINI_API_KEY`, `--real_llm`)
  - Gemini Flash is typically faster than older Claude latencies for short scalar scores — document wall-clock in paper
  - For paper results use real LLM scorer, not mock
- [ ] All RL runs: greedy decode (num_beams=1), max_new_tokens=50
- [ ] All RL runs: N-step reward sampling (every 4 steps) to control overhead
- [ ] All RL runs: long-audio guard (skip generate on batches > 20s audio)

#### Zero-shot Baseline (Required)
- [ ] Run each model with NO fine-tuning on each domain dataset
- [ ] Record WER — this is your "before any adaptation" number
- [ ] Shows why fine-tuning is necessary for domain audio

### 2.4 Evaluation Metrics

#### Primary Metrics
- [ ] **WER (Word Error Rate)** — overall, lower is better
- [ ] **EWER (Entity WER)** — WER computed only on domain terminology
  - For healthcare: medical terms, drug names, anatomy, procedures
  - For VoxPopuli proxy: parliamentary / procedural terms (script uses a dedicated term list)
  - This is your key differentiating metric — the June 2025 paper showed 21% EWER improvement
- [ ] **M-WER (Medical WER)** — for AfriSpeech specifically, already annotated

#### Secondary Metrics
- [ ] **Training time** per epoch (for framework comparison)
- [ ] **GPU memory** peak usage (for practical deployment comparison)
- [ ] **Inference speed** (real-time factor) — how fast does the model transcribe
- [ ] **Params trained** — full fine-tuning vs LoRA
- [ ] **CER** (character error rate), **SER** (sentence error rate)
- [ ] **Domain-term precision / recall / F1** (token-level on domain lexicon)
- [ ] **RL reward trajectory / variance** (logged in results JSON)
- [ ] **Paired bootstrap p-value** (SFT vs RL on same validation refs; script computes when running full pipeline)

#### Evaluation Protocol
- [ ] Evaluate on domain test set (held out, never seen during training)
- [ ] Evaluate on LibriSpeech test-clean (catastrophic forgetting check)
  - If WER on LibriSpeech degrades significantly after domain fine-tuning,
    that's catastrophic forgetting — must be reported
- [ ] Run each experiment minimum 2 times with different seeds, report mean ± std
  - RL training has high variance — single runs are not credible for a paper

### 2.5 Results Table Structure

The paper needs this table (or equivalent):

```
Table 1: Overall WER on domain test sets

Method              | Framework | Healthcare WER | Judicial WER | LibriSpeech WER
--------------------|-----------|----------------|--------------|----------------
Zero-shot           | HF        |                |              |
SFT                 | HF        |                |              |
SFT + RL-MWER       | HF        |                |              |
SFT + RL-WWER       | HF        |                |              |
SFT + RL-LLM        | HF        |                |              |
Zero-shot           | NeMo      |                |              |
SFT                 | NeMo      |                |              |
SFT + RL-MWER       | NeMo      |                |              |
SFT + RL-WWER       | NeMo      |                |              |
SFT + RL-LLM        | NeMo      |                |              |

Table 2: EWER (Entity WER) on domain terminology

Method              | Framework | Healthcare EWER | Judicial EWER
--------------------|-----------|-----------------|---------------
[same rows as above]

Table 3: Efficiency (secondary)

Method              | Framework | Train time/epoch | GPU mem | Params trained
--------------------|-----------|-----------------|---------|---------------
Full fine-tuning    | HF        |                 |         | 244M
LoRA (r=32)         | HF        |                 |         | ~3M
Full fine-tuning    | NeMo      |                 |         | 30M
```

### 2.6 What the Paper Claims (and Must Demonstrate)

1. **SFT → RL improves domain term accuracy** beyond SFT alone
   - Primary evidence: EWER reduction on healthcare and judicial data
   
2. **WWER reward outperforms MWER for domain adaptation**
   - Because it explicitly upweights domain terms in the reward signal
   
3. **LLM-based reward provides additional improvement** over WER-based rewards
   - The novel contribution — connects to the June 2025 paper (21% EWER)
   
4. **Both frameworks (HF and NeMo) support this pipeline**
   - Framework comparison is a secondary contribution
   - Documents implementation effort and performance trade-offs

5. **LoRA is a viable parameter-efficient alternative**
   - Reduces compute cost with minimal WER penalty

---

## PART 3: GCP EXECUTION PLAN

### 3.1 Instance Configuration

```bash
gcloud config set project adaptive-ai-487419

# Recommended: N1 + V100 (matches spot cost estimates; request V100 quota if needed)
gcloud compute instances create stt-nemo-training \
  --project=adaptive-ai-487419 \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# SSH in
gcloud compute ssh stt-nemo-training --zone=us-central1-a --project=adaptive-ai-487419

# Optional: durable artifact storage (checkpoints + results JSON)
gsutil mb -l us-central1 gs://adaptive-ai-487419-stt-results
```

**Cost estimate with SPOT pricing:**
- Per experiment (SFT 5 epochs + 3x RL 2 epochs, 1000 samples): ~$3-5
- Full paper results (2 frameworks × 2 domains × all experiments): ~$25-40
- GCP SPOT is ~70% cheaper than on-demand

### 3.2 Environment Setup (run once after SSH)

```bash
# Install NeMo
pip install nemo_toolkit[asr]
pip install datasets soundfile librosa jiwer peft

# Verify GPU
nvidia-smi  # should show V100 (use python ... --batch_size 8 for conformer-medium on 16GB)

# Login to HuggingFace (needed for some datasets)
huggingface-cli login
# Paste your HF token from https://huggingface.co/settings/tokens
```

### 3.3 Experiment Execution Order

Run experiments in this exact order to minimise cost and maximise usefulness 
if GCP session is interrupted:

```
PHASE 1 — NeMo pipeline (run first)
  1a. NeMo SFT on AfriSpeech clinical (healthcare)      ~45 min  ~$2
  1b. NeMo RL-MWER on healthcare SFT checkpoint         ~20 min  ~$0.80
  1c. NeMo RL-WWER on healthcare SFT checkpoint         ~20 min  ~$0.80
  1d. NeMo SFT on VoxPopuli (judicial)                  ~45 min  ~$2
  1e. NeMo RL-MWER on judicial SFT checkpoint           ~20 min  ~$0.80
  1f. NeMo RL-WWER on judicial SFT checkpoint           ~20 min  ~$0.80
  1g. NeMo zero-shot eval on both domains               ~10 min  ~$0.40

PHASE 2 — HF pipeline
  2a. HF SFT on AfriSpeech clinical                     ~60 min  ~$2.50
  2b. HF RL-MWER on healthcare SFT checkpoint           ~30 min  ~$1.25
  2c. HF RL-WWER on healthcare SFT checkpoint           ~30 min  ~$1.25
  2d. HF SFT on VoxPopuli                               ~60 min  ~$2.50
  2e. HF RL-MWER on judicial SFT checkpoint             ~30 min  ~$1.25
  2f. HF RL-WWER on judicial SFT checkpoint             ~30 min  ~$1.25

PHASE 3 — LLM reward (needs GEMINI_API_KEY + google-generativeai)
  3a. NeMo RL-LLM (real scorer) on both domains         ~60 min  ~$2.50
  3b. HF RL-LLM (real scorer) on both domains           ~60 min  ~$2.50
  Note: Gemini Flash cost is usually minor vs GPU; log tokens/latency in paper

PHASE 4 — LoRA experiments (if time allows)
  4a. HF LoRA SFT on healthcare                         ~30 min  ~$1.25
  4b. HF LoRA + RL-MWER on healthcare                   ~15 min  ~$0.60

PHASE 5 — LibriSpeech catastrophic forgetting check
  5a. Eval all fine-tuned models on LibriSpeech test    ~20 min  ~$0.80
```

**Total estimated cost: $25-35 for full paper results**

### 3.4 Key Config Values for GCP (different from Colab smoke test)

```python
# NeMo GCP config (see gcp_scripts/nemo_afrispeech_training.py CFG)
NEMO_MODEL_NAME = "stt_en_conformer_ctc_medium"
TRAIN_SAMPLES   = None   # None = all clinical train after filter
VAL_SAMPLES     = None   # None = all clinical validation
TEST_SAMPLES    = None   # None = all clinical test
VOXPOPULI_TRAIN_SUBSET = 10000
BATCH_SIZE      = 16
SFT_EPOCHS      = 5
RL_EPOCHS       = 2
SFT_LR          = 1e-4
RL_LR           = 1e-5
REWARD_WEIGHT   = 0.05
REWARD_STEP_INTERVAL = 4
MAX_ENCODER_LEN_FOR_REWARD = 2000
num_workers     = 4

# HF GCP config
MODEL_NAME      = "openai/whisper-small"          # was whisper-base
TRAIN_SAMPLES   = 1000
EVAL_SAMPLES    = 200
BATCH_SIZE      = 16
SFT_EPOCHS      = 5
RL_EPOCHS       = 2
SFT_LR          = 1e-5
RL_LR           = 1e-6
RL_REWARD_WEIGHT= 0.05
num_proc        = 4                               # was 1
dataloader_num_workers = 4                        # was 2
```

### 3.5 Running in Background (essential — prevents loss on disconnect)

```bash
# Upload script
gcloud compute scp gcp_scripts/nemo_afrispeech_training.py stt-nemo-training:~/nemo-stt/ \
  --zone=us-central1-a --project=adaptive-ai-487419

# Run in background with logging (add --upload_gcs gs://... for durable saves)
cd ~/nemo-stt
nohup python nemo_afrispeech_training.py --stage both \
  --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/run_$(date +%Y%m%d_%H%M) \
  > nemo_training.log 2>&1 &

# Watch log live (in a separate terminal)
gcloud compute ssh stt-nemo-training --zone=us-central1-a
tail -f nemo_training.log

# Check it's running
ps aux | grep python
```

### 3.6 Saving Results Before Instance Stops

```bash
# From your local machine — download all checkpoints + logs
gcloud compute scp --recurse \
  stt-nemo-training:~/nemo-sft-stage/ ./results/nemo-sft/ 
gcloud compute scp --recurse \
  stt-nemo-training:~/nemo-rl-stage/ ./results/nemo-rl/
gcloud compute scp stt-nemo-training:~/nemo_training.log ./results/

# ALWAYS stop instance after each session
gcloud compute instances stop stt-nemo-training --zone=us-central1-a

# Verify nothing is still running (avoid surprise bills)
gcloud compute instances list
```

---

## PART 4: CODE REQUIREMENTS CHECKLIST (NeMo script)

The NeMo GCP script (`gcp_scripts/nemo_afrispeech_training.py`) must provide:

### Data
- [ ] `load_afrispeech_medical()` — primary domain dataset, filters to clinical
- [ ] `load_voxpopuli_judicial()` — judicial domain dataset
- [ ] `load_librispeech()` — general baseline
- [ ] `load_dataset_with_fallback()` — tries in priority order
- [ ] `build_nemo_manifest()` — converts HF dataset to NeMo JSON manifest
  - Must skip audio > 20s (prevents generate() hangs)
  - Must lowercase text (NeMo CTC requirement)

### Model
- [ ] `NemoRewardModel` subclassing `EncDecCTCModelBPE`
- [ ] `from_pretrained_with_reward()` — for Stage 1 (SFT)
- [ ] `from_checkpoint_with_reward()` — for Stage 2 (RL from SFT checkpoint)
- [ ] `_compute_loss()` override with:
  - `reward_mode="none"` → pure CTC, returns immediately
  - `reward_mode="mwer"/"wwer"/"llm"/"all"` → reward injection
  - N-step sampling (every 4 steps) — reduces overhead
  - Long-audio guard (skip generate if input_lengths.max() > 2000)
  - Greedy CTC decode (argmax) not beam search
  - Cached last hypotheses for steps where generate is skipped

### Training
- [ ] `run_sft_stage()` — Stage 1, pure CTC, saves checkpoint
- [ ] `run_rl_stage()` — Stage 2, loads SFT checkpoint, applies reward
- [ ] Warmup steps computed automatically as 10% of total steps
  - CRITICAL: warmup_steps must always be < total_steps
  - total_steps = ceil(TRAIN_SAMPLES / BATCH_SIZE) * NUM_EPOCHS
- [ ] `log_every_n_steps=10` (not 25 — too infrequent for 1000 samples)
- [ ] `NemoTrainingLogger` callback logging loss + LR every 10 steps

### Evaluation
- [ ] `evaluate_wer()` — overall WER using jiwer
- [ ] `evaluate_ewer()` — entity WER on domain term list
  - Healthcare terms: myocardial, infarction, hypertension, tachycardia,
    arrhythmia, stethoscope, auscultation, echocardiogram, medications list
  - Judicial terms: plaintiff, defendant, affidavit, subpoena,
    jurisdiction, deposition, mandamus, injunction, legal Latin
- [ ] `evaluate_zero_shot()` — run model before any fine-tuning
- [ ] `evaluate_catastrophic_forgetting()` — run on LibriSpeech after fine-tuning

### Results
- [ ] All results saved to a structured JSON file at the end
- [ ] Printed comparison table: zero-shot vs SFT vs each RL method
- [ ] Per-epoch metrics logged to separate CSV for plotting

### Reward Functions (same as HF version, framework-agnostic)
- [ ] `compute_mwer_reward()` — reward = 1 - WER
- [ ] `compute_wwer_reward()` — domain terms weighted 3x
- [ ] `compute_llm_reward()` — mock (default) or real via **Gemini API** (`GEMINI_API_KEY`, `--real_llm`)
- [ ] `test_reward_functions()` — sanity check before training

---

## PART 5: KNOWN BUGS FIXED (don't reintroduce these)

These bugs were debugged in Colab and must not appear in the GCP version:

1. **warmup_steps > total_steps** — causes LR to never converge. 
   Fix: compute warmup as 10% of total_steps, always.

2. **reward_weight=0.3 default** — too high, fights CTC loss, WER gets worse.
   Fix: always use 0.05.

3. **RL from scratch (no SFT first)** — reward signal is noise when model 
   can't transcribe. Always load SFT checkpoint for Stage 2.

4. **generate() deadlock on long audio** — beam search on audio > 20s hangs.
   Fix: skip generate() on batches with max input_length > 2000 frames.

5. **N-step reward not implemented** — calling generate() every step is 4x 
   slower than needed. Fix: only compute reward every 4 steps, cache result.

6. **trust_remote_code deprecation** — newer HF datasets library doesn't 
   support this flag. Remove it from all dataset load calls.

7. **NotebookProgressCallback crash** — only in Jupyter/Colab, not on GCP 
   (plain Python). Still safe to call remove_callback() defensively.

---

## PART 6: WHAT THIS PAPER LOOKS LIKE WHEN DONE

### Abstract claim (target)
"We present the first systematic comparison of reinforcement learning reward 
designs for domain-specific speech recognition, evaluated on healthcare and 
judicial audio. Using a two-stage SFT→RL pipeline across HuggingFace 
Transformers and NVIDIA NeMo, we show that WWER reward signals achieve 
[X]% relative EWER improvement over SFT on clinical speech, with LLM-based 
rewards providing an additional [Y]% gain. We provide reward design 
principles and an open-source implementation supporting both frameworks."

### Minimum viable paper (if time is tight)
1. SFT baseline on AfriSpeech clinical (healthcare only)
2. RL-MWER and RL-WWER on top of SFT
3. Comparison table: SFT vs RL-MWER vs RL-WWER
4. One framework only (NeMo) is acceptable for a workshop paper

### Full paper
All of the above plus judicial domain, LLM reward, framework comparison, 
LoRA, catastrophic forgetting analysis.
