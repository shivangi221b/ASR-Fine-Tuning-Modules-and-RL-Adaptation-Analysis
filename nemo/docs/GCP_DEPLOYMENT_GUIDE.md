# GCP Deployment Guide for NeMo STT Training

This guide covers running [`gcp_scripts/nemo_afrispeech_training.py`](../gcp_scripts/nemo_afrispeech_training.py) on Google Cloud with GPUs.

**Default GCP project (this repo):** `adaptive-ai-487419`  
**Suggested results bucket:** `gs://adaptive-ai-487419-stt-results`

---

## Overview

**Recommended setup**

- Instance: `n1-standard-8` (8 vCPU, 30GB RAM)
- GPU: 1× NVIDIA A100 40GB (preferred) or 1× V100 16GB
- Boot disk: 200GB SSD
- Spot / preemptible: yes, for cost (~70% savings vs on-demand)

**Training times (rough)**

- Local/GPU smoke test: `python nemo_afrispeech_training.py --smoke_test` → a few minutes
- Full AfriSpeech clinical train (~36k clips) + SFT 5 epochs + RL: several GPU-hours (see project cost notes in `project_summary_and_gcp_plan.md`)

---

## One-time: GCS bucket for checkpoints + JSON results

Results and `.nemo` files can be uploaded automatically if `gsutil` is available and the VM has storage scope.

```bash
gcloud config set project adaptive-ai-487419

# Create bucket (location near your VM)
gsutil mb -l us-central1 gs://adaptive-ai-487419-stt-results

# Ensure the training service account / VM can write (Editor on bucket or roles/storage.objectAdmin)
```

Pass `--upload_gcs gs://adaptive-ai-487419-stt-results/run_name` to the training script; it will `gsutil cp` checkpoints and `results/*_results.json` after each stage.

---

## Option 1: Compute Engine VM (recommended)

### Create VM

```bash
gcloud config set project adaptive-ai-487419

gcloud compute instances create nemo-training \
    --zone=us-central1-a \
    --project=adaptive-ai-487419 \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --preemptible
```

### SSH

```bash
gcloud compute ssh nemo-training --zone=us-central1-a --project=adaptive-ai-487419
```

### Environment

```bash
mkdir -p ~/nemo-stt && cd ~/nemo-stt

pip install -U pip
pip install "nemo_toolkit[asr]" datasets soundfile librosa jiwer pandas numpy google-generativeai

# Optional: Gemini for RL-LLM (export on VM or use --machine metadata / secret manager)
export GEMINI_API_KEY="..."

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Upload script

From your laptop:

```bash
gcloud compute scp gcp_scripts/nemo_afrispeech_training.py \
  nemo-training:~/nemo-stt/ --zone=us-central1-a --project=adaptive-ai-487419
```

### Run

**Smoke test (crash check before spending GPU time):**

```bash
cd ~/nemo-stt
python nemo_afrispeech_training.py --smoke_test
```

**Full AfriSpeech clinical (all train clips, official val/test splits) + GCS upload:**

```bash
tmux new -s train
python nemo_afrispeech_training.py --stage both \
  --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/exp_afrispeech_$(date +%Y%m%d) \
  --seed 42
```

**SFT only, then RL in separate jobs:**

```bash
python nemo_afrispeech_training.py --stage sft --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/exp1

python nemo_afrispeech_training.py --stage rl \
  --sft_checkpoint ./checkpoints/sft_model.nemo \
  --reward_mode wwer \
  --upload_gcs gs://adaptive-ai-487419-stt-results/exp1
```

**VoxPopuli (random 10k train, official validation — fixed seed in script):**

```bash
python nemo_afrispeech_training.py --stage both --dataset voxpopuli \
  --voxpopuli_train_subset 10000 \
  --upload_gcs gs://adaptive-ai-487419-stt-results/exp_vox
```

**Real Gemini LLM reward (not mock):**

```bash
export GEMINI_API_KEY="..."
python nemo_afrispeech_training.py --stage rl \
  --sft_checkpoint ./checkpoints/sft_model.nemo \
  --reward_mode llm --real_llm
```

### Download results (if not using GCS)

```bash
gcloud compute scp --recurse \
  nemo-training:~/nemo-stt/checkpoints ./results/ \
  --zone=us-central1-a --project=adaptive-ai-487419
gcloud compute scp --recurse \
  nemo-training:~/nemo-stt/results ./results_metrics/ \
  --zone=us-central1-a --project=adaptive-ai-487419
```

### Stop VM (avoid idle charges)

```bash
gcloud compute instances stop nemo-training --zone=us-central1-a --project=adaptive-ai-487419
```

---

## Option 2: Vertex AI Custom Training

See historical example in git history or adapt: container / prebuilt PyTorch DL image, `pip install` deps, run the same CLI as above. Ensure `GEMINI_API_KEY` is injected via environment or Secret Manager.

---

## Multi-dataset paper workflow

```bash
# AfriSpeech clinical (default caps: full train, full val, full test clinical)
python nemo_afrispeech_training.py --stage both --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/paper/afrispeech

mv checkpoints/sft_model.nemo checkpoints/sft_afrispeech.nemo
mv checkpoints/rl_model.nemo checkpoints/rl_afrispeech_mwer.nemo  # rename per reward run

# VoxPopuli
python nemo_afrispeech_training.py --stage both --dataset voxpopuli \
  --voxpopuli_train_subset 10000 \
  --upload_gcs gs://adaptive-ai-487419-stt-results/paper/voxpopuli
```

---

## Cost reference

| Setup | Spot $/hr (approx.) |
|-------|------------------------|
| n1-standard-8 + A100 | ~$1.50 |
| n1-standard-8 + V100 | ~$0.80 |

**GCS:** ~$0.020/GB-month standard; keep `.nemo` + JSON only for low cost.

---

## Troubleshooting

- **OOM:** reduce batch size in `Config` / add CLI override (future) or edit `BATCH_SIZE` in script.
- **Preemption:** rely on GCS uploads per stage; restart VM and continue from saved `.nemo`.
- **AfriSpeech streaming slow:** first streaming pass filters clinical; manifests cache WAVs under `OUTPUT_DIR/audio`.
- **`gsutil` not found:** install Google Cloud SDK or use `gcloud storage cp` (newer).
