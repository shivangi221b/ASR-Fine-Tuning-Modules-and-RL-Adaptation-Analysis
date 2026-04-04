# GCP Deployment Guide for NeMo STT Training

This guide covers running [`gcp_scripts/nemo_afrispeech_training.py`](../gcp_scripts/nemo_afrispeech_training.py) on Google Cloud with GPUs.

**Console (sign-in):** [Google Cloud Console — project adaptive-ai-487419](https://console.cloud.google.com/welcome?project=adaptive-ai-487419)

**Default GCP project (this repo):** `adaptive-ai-487419`  
**Suggested results bucket:** `gs://adaptive-ai-487419-stt-results`

---

## Onboarding checklist (one-time on your laptop)

1. **Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)** (`gcloud`, `gsutil`).
2. **Authenticate:** `gcloud auth login` (browser) and, for VM SSH later, `gcloud auth application-default login` if needed.
3. **Select project:** `gcloud config set project adaptive-ai-487419`
4. **Billing:** In Cloud Console → Billing, link a billing account to this project (GPU VMs will not start without it).
5. **Enable APIs** (Console → APIs & Services, or run once):
   - Compute Engine API  
   - (Optional) Cloud Storage if prompted when creating buckets.
6. **GPU quota:** Console → IAM & Admin → Quotas — ensure **NVIDIA T4** (recommended) or **V100** quota in your zone for N1+GPU; use **A2 / A100** quota only if you pick `a2-highgpu-1g`. Request increases if quota is 0.
7. **Hugging Face token** (for some datasets): on the VM, `huggingface-cli login` after `pip install huggingface_hub`.

---

## Your laptop can sleep — training runs on the VM

Training **does not** run on your Mac. It runs **inside the GCP VM** after you SSH in.

- **Closing the laptop or losing Wi‑Fi** only drops your **SSH session**. It does **not** stop the VM by default.
- To keep the **Python process** running after disconnect, start training inside **`tmux`** or **`screen`**, or use **`nohup`**:
  - **tmux (recommended):** `tmux new -s train` → run your `python ...` command → press `Ctrl+B`, then `D` to detach. Reattach later: `tmux attach -t train`.
  - **nohup:** `nohup python ... > train.log 2>&1 &` then `tail -f train.log`.

Always pass **`--upload_gcs gs://.../run_id`** so checkpoints and `*_results.json` are copied to Cloud Storage even if the VM is preempted (Spot) or you delete the instance later.

---

## Overview

**Recommended setup**

- **`n1-standard-8` + 1× T4** — **Best match for PyTorch 2.x on Deep Learning images:** T4 is compute capability **7.5**, which current wheels support; V100 is **7.0** and triggers “GPU too old” warnings with PyTorch 2.7+cu128. T4 is also **usually cheaper per hour** than V100 (see cost table), but **slower** than V100 for the same batch size. Use **`--batch_size 4`** or **`8`** for `stt_en_conformer_ctc_medium` on 16GB.
- **Optional — V100:** faster than T4, but you should **pin an older PyTorch/CUDA** that still supports sm_70 or accept unsupported-toolchain risk.
- **Optional — A100:** `a2-highgpu-1g` only (A100 cannot attach to N1). Fastest; higher $/hr. Do **not** use `--accelerator=nvidia-tesla-a100` on `n1-standard-*`.
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

# Image family: Google retired the meta-family `pytorch-latest-gpu`. List current
# families with:
#   gcloud compute images list --project deeplearning-platform-release \
#     --no-standard-images --filter="family~pytorch" --format="value(family)" | sort -u
gcloud compute instances create nemo-training \
    --zone=us-central1-a \
    --project=adaptive-ai-487419 \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --preemptible

# Faster GPU (A100): use --machine-type=a2-highgpu-1g and remove --accelerator (GPU is built-in).
# Optional: V100 — --accelerator=type=nvidia-tesla-v100,count=1 (see Overview: PyTorch 2.7 vs V100).
#
# Do not set install-nvidia-driver=True on PyTorch DL images (family names containing
# nvidia-XXX): the driver is already matched to CUDA on the image; the metadata installer
# can conflict and break nvidia-smi until you reboot or repair. Use install-nvidia-driver
# only on plain Ubuntu/Debian images without a preinstalled driver.
```

### SSH

```bash
gcloud compute ssh nemo-training --zone=us-central1-a --project=adaptive-ai-487419
```

**`Connection refused` on port 22 right after create:** normal for **several minutes** while the guest OS finishes booting. Wait **5–15 minutes**, then run `gcloud compute ssh` again. To see boot progress:

```bash
gcloud compute instances get-serial-port-output nemo-training \
  --zone=us-central1-a --project=adaptive-ai-487419 | tail -80
```

If it stays refused after ~20 minutes, check **VPC → Firewall** that a rule allows **tcp:22** to instances with the default network tag (usually `default-allow-ssh`).

**`External IP address was not found` / IAP tunnel errors:** Usually the VM is **stopped** (`TERMINATED`). Ephemeral public IPs are dropped when the instance stops, so `gcloud` may fall back to IAP and fail. **Start** the instance, wait until `RUNNING`, then SSH:

```bash
gcloud compute instances start nemo-training-1 --zone=us-central1-a --project=adaptive-ai-487419
gcloud compute instances describe nemo-training-1 --zone=us-central1-a --format='get(status)'
gcloud compute ssh nemo-training-1 --zone=us-central1-a --project=adaptive-ai-487419
```

### Environment

Use the **same pins as local** (NumPy &lt; 2, datasets &lt; 3) to avoid NeMo / AfriSpeech breakage. From your laptop you can copy the whole repo or only the requirements file.

```bash
mkdir -p ~/nemo-stt && cd ~/nemo-stt

# Option A: copy requirements from your machine (paths relative to your laptop)
gcloud compute scp requirements-nemo-train.txt nemo-training:~/nemo-stt/ \
  --zone=us-central1-a --project=adaptive-ai-487419

# On the VM:
pip install -U pip
pip install -r requirements-nemo-train.txt

# Option B: quick one-liner (may drift from repo pins — prefer Option A)
# pip install -U pip && pip install "numpy>=1.22,<2.0" "datasets>=2.14.0,<3.0.0" \
#   "nemo_toolkit[asr]" lightning torch soundfile librosa jiwer pandas omegaconf google-generativeai

# Optional: Gemini for RL-LLM
export GEMINI_API_KEY="..."

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Full runs on 16GB (T4/V100): use smaller batches (script flag or edit Config)
#   python nemo_afrispeech_training.py --batch_size 8 --upload_gcs gs://.../run1
```

### Upload script (and optional full `gcp_scripts/`)

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

**Full AfriSpeech clinical (all train clips, official val/test splits) + GCS upload**  
(Run inside `tmux` so you can close your laptop; see section above.)

```bash
cd ~/nemo-stt
tmux new -s train
python nemo_afrispeech_training.py --stage both \
  --dataset afrispeech_clinical \
  --upload_gcs gs://adaptive-ai-487419-stt-results/exp_afrispeech_$(date +%Y%m%d_%H%M) \
  --seed 42
# Detach: Ctrl+B, then D
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

The script runs **one** `REWARD_MODE` per invocation (`mwer`, `wwer`, `llm`, or `all`). For tables in the paper:

1. Run **SFT once** (or reuse `sft_model.nemo`).
2. Run **RL** separately for **mwer**, **wwer**, and **llm** (`--real_llm` + `GEMINI_API_KEY`), using `--sft_checkpoint` for WWER/LLM if you already have SFT, with distinct `--upload_gcs` paths or rename `.nemo` after each run.
3. Repeat with a **second `--seed`** for mean ± std.

```bash
# Example: AfriSpeech full pipeline with MWER
python nemo_afrispeech_training.py --stage both --dataset afrispeech_clinical \
  --reward_mode mwer \
  --upload_gcs gs://adaptive-ai-487419-stt-results/paper/afrispeech_mwer_seed42

# WWER only (reuse SFT checkpoint)
python nemo_afrispeech_training.py --stage rl --dataset afrispeech_clinical \
  --sft_checkpoint ./checkpoints/sft_model.nemo \
  --reward_mode wwer \
  --upload_gcs gs://adaptive-ai-487419-stt-results/paper/afrispeech_wwer_seed42

# VoxPopuli
python nemo_afrispeech_training.py --stage both --dataset voxpopuli \
  --voxpopuli_train_subset 10000 \
  --upload_gcs gs://adaptive-ai-487419-stt-results/paper/voxpopuli_mwer_seed42
```

---

## Paper coverage vs this NeMo script

**In scope for this file:** AfriSpeech / VoxPopuli / Libri forget; zero-shot; SFT + reward stage; metrics (WER, CER, SER, EWER-style, domain F1); bootstrap on val; test split eval; GCS persistence; Gemini LLM reward.

**Outside this file:** HuggingFace Whisper pipeline (second framework); AfriSpeech **M-WER / M-CER** from gold entity spans (not wired); automatic multi-seed aggregation; peak GPU memory / RTF in JSON; “best val WER” checkpoint selection (currently last epoch); narrative comparison to published AfriSpeech baselines (your writing).

---

## Cost reference

| Setup | Spot $/hr (approx.) |
|-------|------------------------|
| a2-highgpu-1g (1× A100 40GB) | ~$2–4 (region/pricing varies) |
| n1-standard-8 + V100 | ~$0.80 |
| n1-standard-8 + T4 | ~$0.45–0.65 (often **below** V100; confirm in your region) |

**GCS:** ~$0.020/GB-month standard; keep `.nemo` + JSON only for low cost.

---

## Troubleshooting

- **`n1-standard-*` + A100 “not compatible”:** switch to `a2-highgpu-1g` and drop `--accelerator` (A100 is tied to A2 shapes). Or keep N1 and use V100/T4.
- **PyTorch “V100 / cuda capability 7.0 … minimum is 7.5”:** PyTorch 2.7 wheels target newer GPUs. Prefer **`nvidia-tesla-t4`** (or A100) on N1, or pin an older PyTorch + CUDA that still supports Volta.
- **OOM:** reduce batch size in `Config` / add CLI override (future) or edit `BATCH_SIZE` in script.
- **Preemption:** rely on GCS uploads per stage; restart VM and continue from saved `.nemo`.
- **AfriSpeech streaming slow:** first streaming pass filters clinical; manifests cache WAVs under `OUTPUT_DIR/audio`.
- **`gsutil` not found:** install Google Cloud SDK or use `gcloud storage cp` (newer).
- **SSH `Connection refused`:** wait for first boot (see SSH section); use serial port output above. Not the same as “permission denied” (keys) or timeout (firewall / wrong IP).
- **`nvidia-smi` “couldn't communicate with the NVIDIA driver”:** On PyTorch DL images, **omit `install-nvidia-driver=True`** when creating the VM (see create command). If **`lspci | grep -i nvidia`** shows the GPU but **`nvidia-smi`** still fails after reboot + headers, the kernel module is not loaded — use **Google’s installer** (below). Optional diagnostics: `sudo modprobe nvidia` (read the error), `dkms status`, `dmesg | grep -i nvidia`.
- **Boot disk larger than image (e.g. 200GB):** Ubuntu on GCP usually expands the root filesystem on first boot; if `df -h` still shows ~100GB, follow [resize persistent disk](https://cloud.google.com/compute/docs/disks/add-persistent-disk#resize_pd) / grow the partition once.

### GPU driver repair (`lspci` shows NVIDIA, `nvidia-smi` fails)

Google’s supported fix is **`cuda_installer.pyz`** ([install script](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#install-script)). On the VM:

```bash
# If Ops Agent is installed, stop it first (can block driver install).
sudo systemctl stop google-cloud-ops-agent 2>/dev/null || true

cd /tmp
curl -fsSL -o cuda_installer.pyz https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod
```

The installer may **reboot** the VM once or twice; if the docs say to re-run after restart, run the same `install_driver` line again, then **`sudo reboot`** once more and test `nvidia-smi`. PyTorch/CUDA on the DL image should remain usable; you are repairing the **kernel driver** only.

**If `cuda_installer` still leaves you broken** — especially when:

`modprobe: FATAL: Module nvidia not found in directory /lib/modules/$(uname -r)` and **`dkms status`** is empty — the running **GCP kernel updated** (e.g. `6.8.0-1053-gcp`) but **no NVIDIA module exists for that kernel**. Install a driver through **apt** so **DKMS** builds the module for the current kernel:

```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) build-essential dkms ubuntu-drivers-common
# DL VM images sometimes omit ubuntu-drivers; the package above provides `ubuntu-drivers`.
sudo ubuntu-drivers list
sudo ubuntu-drivers autoinstall
# If autoinstall fails or recommends nothing useful for Tesla, install explicitly:
# sudo apt-get install -y nvidia-driver-550-server
sudo reboot
```

After reboot, **`nvidia-smi`** should work. If **`ubuntu-drivers list`** shows nothing useful, install **`nvidia-driver-550-server`** or **`nvidia-driver-535-server`** explicitly (Tesla V100 is supported by the server/meta packages on Ubuntu 22.04). To see available packages: `apt-cache search ^nvidia-driver- | grep server`.

As a last resort, **recreate** the VM **without** `install-nvidia-driver` metadata and avoid `apt full-upgrade` before confirming `nvidia-smi`, or **hold** the kernel package until the driver stack is healthy (advanced).

If that still fails, capture **`sudo modprobe nvidia`**, **`dkms status`**, and **`dpkg -l | grep nvidia`** output for support.
