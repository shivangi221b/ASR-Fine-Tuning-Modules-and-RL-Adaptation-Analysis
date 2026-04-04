"""
NeMo STT Domain Adaptation: SFT + Reward-Weighted Fine-Tuning
===============================================================

Two-stage training:
  - Stage 1: Supervised Fine-Tuning (SFT) with CTC loss
  - Stage 2: Reward-augmented fine-tuning on top of SFT (MWER / WWER / LLM rewards)

Datasets:
  - AfriSpeech-200 clinical (official train / validation / test splits)
  - VoxPopuli English (random train subset + official validation)
  - LibriSpeech clean-100 (optional train; full validation for forgetting checks)

Usage:
  python nemo_afrispeech_training.py --smoke_test
  python nemo_afrispeech_training.py --stage sft --dataset afrispeech_clinical
  python nemo_afrispeech_training.py --stage rl --sft_checkpoint ./checkpoints/sft_model.nemo
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from jiwer import cer as compute_cer_jiwer
from jiwer import wer as compute_wer_jiwer
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Repo root and shared data paths
# ---------------------------------------------------------------------------
# parents[0] = gcp_scripts/, parents[1] = nemo/, parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_MANIFEST_DIR = str(_REPO_ROOT / "data" / "manifests")
_SHARED_AUDIO_DIR = str(_REPO_ROOT / "data" / "audio")

sys.path.insert(0, str(_REPO_ROOT))
from data import (  # noqa: E402
    build_nemo_manifest,
    load_dataset_bundle,
    load_librispeech_eval,
)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from lightning.pytorch import LightningModule
from nemo.collections.asr.models import EncDecCTCModelBPE


def _require_datasets_script_support() -> None:
    """HuggingFace `datasets` 3+ no longer runs hub dataset .py scripts (e.g. AfriSpeech)."""
    parts = _datasets_lib.__version__.split(".")
    major = int(parts[0]) if parts[0].isdigit() else 0
    if major >= 3:
        raise RuntimeError(
            f"AfriSpeech-200 needs `datasets` < 3 (dataset scripts). You have {_datasets_lib.__version__}. "
            'Run: python -m pip install "datasets>=2.14.0,<3.0.0"'
        )


# ---------------------------------------------------------------------------
# Optional Gemini (LLM reward)
# ---------------------------------------------------------------------------
try:
    import google.generativeai as genai

    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False
    genai = None  # type: ignore

# ===========================================================================
# CONFIGURATION
# ===========================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration (mutated by CLI / smoke test)."""

    NEMO_MODEL_NAME: str = "stt_en_conformer_ctc_medium"
    DATASET: str = "afrispeech_clinical"  # afrispeech_clinical | librispeech | voxpopuli

    # AfriSpeech: None = all available clinical in split (after filter)
    TRAIN_SAMPLES: Optional[int] = None  # cap train size; None = full clinical train
    VAL_SAMPLES: Optional[int] = None  # cap val; None = all clinical validation
    TEST_SAMPLES: Optional[int] = None  # cap test; None = all clinical test

    # VoxPopuli random train subset
    VOXPOPULI_TRAIN_SUBSET: int = 10_000

    # LibriSpeech (baseline / forgetting): train cap, eval uses full val split slice
    LIBRISPEECH_TRAIN_CAP: int = 5_000

    BATCH_SIZE: int = 16
    LEARNING_RATE_SFT: float = 1e-4
    LEARNING_RATE_RL: float = 1e-5  # 1/10 SFT per paper plan

    SFT_EPOCHS: int = 5
    RL_EPOCHS: int = 2

    REWARD_MODE: str = "mwer"  # mwer | wwer | llm | all
    REWARD_WEIGHT: float = 0.05
    REWARD_STEP_INTERVAL: int = 4  # inject reward every N optimizer steps
    MAX_ENCODER_LEN_FOR_REWARD: int = 2000  # long-audio guard (frames)

    OUTPUT_DIR: str = "./nemo-afrispeech-output"  # training artifacts (checkpoints, results)
    CHECKPOINT_DIR: str = "./checkpoints"
    RESULTS_DIR: str = "./results"
    # Manifests and audio are written to the shared data/ directory (see _SHARED_MANIFEST_DIR
    # and _SHARED_AUDIO_DIR) and are not configurable here to keep them toolkit-agnostic.

    SEED: int = 42

    # Healthcare (AfriSpeech) domain terms for WWER / EWER
    DOMAIN_TERMS_CLINICAL: frozenset = field(
        default_factory=lambda: frozenset(
            {
                "hypertension",
                "diabetes",
                "malaria",
                "fever",
                "headache",
                "medication",
                "prescription",
                "diagnosis",
                "symptoms",
                "patient",
                "doctor",
                "clinic",
                "hospital",
                "treatment",
                "blood",
                "pressure",
                "temperature",
                "pregnant",
                "pregnancy",
                "vaccine",
                "injection",
                "myocardial",
                "infarction",
                "tachycardia",
                "arrhythmia",
                "stethoscope",
                "auscultation",
                "echocardiogram",
                "tuberculosis",
                "antiretroviral",
                "paracetamol",
                "ibuprofen",
                "amoxicillin",
            }
        )
    )
    # Formal / parliamentary vocabulary (VoxPopuli proxy)
    DOMAIN_TERMS_PARLIAMENTARY: frozenset = field(
        default_factory=lambda: frozenset(
            {
                "parliament",
                "commission",
                "directive",
                "regulation",
                "subsidiarity",
                "rapporteur",
                "plenaries",
                "legislation",
                "mandate",
                "quorum",
                "amendment",
                "plenaries",
                "council",
                "plenary",
            }
        )
    )
    DOMAIN_TERM_WEIGHT: float = 3.0

    GEMINI_MODEL: str = "gemini-1.5-flash"
    USE_MOCK_LLM: bool = True

    SMOKE_TEST: bool = False
    SKIP_GCS: bool = False
    UPLOAD_GCS_URI: Optional[str] = None  # e.g. gs://adaptive-ai-487419-stt-results/run1

    USE_LORA: bool = False

    RUN_ZERO_SHOT: bool = True
    RUN_LIBRISPEECH_FORGETTING: bool = True
    RUN_FINAL_TEST_EVAL: bool = True
    BOOTSTRAP_ITERS: int = 1000


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_domain_terms() -> frozenset:
    if CFG.DATASET == "voxpopuli":
        return CFG.DOMAIN_TERMS_PARLIAMENTARY
    return CFG.DOMAIN_TERMS_CLINICAL


def apply_smoke_test_overrides() -> None:
    CFG.SMOKE_TEST = True
    CFG.TRAIN_SAMPLES = 30
    CFG.VAL_SAMPLES = 10
    CFG.TEST_SAMPLES = 5
    CFG.VOXPOPULI_TRAIN_SUBSET = 40
    CFG.LIBRISPEECH_TRAIN_CAP = 20
    CFG.BATCH_SIZE = 2
    CFG.SFT_EPOCHS = 1
    CFG.RL_EPOCHS = 1
    CFG.NEMO_MODEL_NAME = "stt_en_conformer_ctc_small"
    CFG.USE_MOCK_LLM = True
    CFG.SKIP_GCS = True
    CFG.RUN_LIBRISPEECH_FORGETTING = False
    CFG.BOOTSTRAP_ITERS = 100
    CFG.RUN_FINAL_TEST_EVAL = True


# ===========================================================================
# ALIGNMENT-BASED WWER / weighted error
# ===========================================================================


def _word_err_alignment_cost(ref_words: List[str], hyp_words: List[str], weights: List[float]) -> float:
    """Minimum weighted word error via DP alignment (Levenshtein on words)."""
    R, H = len(ref_words), len(hyp_words)
    inf = float("inf")
    dp = [[inf] * (H + 1) for _ in range(R + 1)]
    dp[0][0] = 0.0
    for i in range(1, R + 1):
        dp[i][0] = dp[i - 1][0] + weights[i - 1]
    for j in range(1, H + 1):
        dp[0][j] = dp[0][j - 1] + 1.0  # insertion
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            w_del = weights[i - 1]
            w_ins = 1.0
            if ref_words[i - 1] == hyp_words[j - 1]:
                sub = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + w_del
            dp[i][j] = min(dp[i - 1][j] + w_del, dp[i][j - 1] + w_ins, sub)
    denom = float(sum(weights)) if R > 0 else 1.0
    return dp[R][H] / denom


def weighted_wer_rate(ref: str, hyp: str, domain: frozenset, w_domain: float) -> float:
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    weights = [w_domain if rw in domain else 1.0 for rw in ref_words]
    return min(1.0, _word_err_alignment_cost(ref_words, hyp_words, weights))


def compute_wwer_reward(hypotheses: List[str], references: List[str]) -> torch.Tensor:
    dom = current_domain_terms()
    rewards = []
    for hyp, ref in zip(hypotheses, references):
        if not ref.strip():
            rewards.append(0.5)
            continue
        rate = weighted_wer_rate(ref, hyp, dom, CFG.DOMAIN_TERM_WEIGHT)
        rewards.append(max(0.0, 1.0 - rate))
    return torch.tensor(rewards, dtype=torch.float32)


def compute_mwer_reward(hypotheses: List[str], references: List[str]) -> torch.Tensor:
    rewards = []
    for hyp, ref in zip(hypotheses, references):
        if not ref.strip():
            rewards.append(0.5)
            continue
        err = compute_wer_jiwer(ref, hyp)
        rewards.append(max(0.0, 1.0 - err))
    return torch.tensor(rewards, dtype=torch.float32)


def compute_llm_reward(
    hypotheses: List[str],
    references: List[str],
    use_mock: Optional[bool] = None,
) -> torch.Tensor:
    if use_mock is None:
        use_mock = CFG.USE_MOCK_LLM or CFG.SMOKE_TEST
    if use_mock or not _HAS_GENAI:
        base = compute_mwer_reward(hypotheses, references)
        noise = torch.randn_like(base) * 0.02
        return torch.clamp(base + noise, 0.0, 1.0)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using MWER proxy for LLM reward")
        return compute_mwer_reward(hypotheses, references)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(CFG.GEMINI_MODEL)
    try:
        gen_cfg = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=16)
    except Exception:
        gen_cfg = {"temperature": 0.0, "max_output_tokens": 16}
    rewards: List[float] = []
    prompt_sys = (
        "You score automatic speech recognition quality for domain-specific English. "
        "Output exactly one number between 0.0 and 1.0 (inclusive), no other text."
    )
    for hyp, ref in zip(hypotheses, references):
        user = f"Reference:\n{ref}\nHypothesis:\n{hyp}\nScore:"
        try:
            resp = model.generate_content([prompt_sys, user], generation_config=gen_cfg)
            txt = (resp.text or "").strip()
            val = float(txt.split()[0])
            val = max(0.0, min(1.0, val))
            rewards.append(val)
        except Exception as e:
            logger.debug("Gemini reward fallback: %s", e)
            rewards.append(float(compute_mwer_reward([hyp], [ref])[0].item()))
        time.sleep(0.05)  # light rate limiting
    return torch.tensor(rewards, dtype=torch.float32)


def compute_combined_reward(hypotheses: List[str], references: List[str]) -> torch.Tensor:
    r1 = compute_mwer_reward(hypotheses, references)
    r2 = compute_wwer_reward(hypotheses, references)
    r3 = compute_llm_reward(hypotheses, references)
    return (r1 + r2 + r3) / 3.0


def test_reward_functions() -> None:
    h = ["the patient has hypertension"]
    r = ["the patient has hypertension"]
    assert float(compute_mwer_reward(h, r)[0]) > 0.95
    for tensor in (compute_mwer_reward(h, h), compute_wwer_reward(h, h)):
        assert 0.0 <= float(tensor[0]) <= 1.0


# ===========================================================================
# EVALUATION METRICS
# ===========================================================================


def _normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def entity_wer_from_text(refs: Sequence[str], hyps: Sequence[str], domain: frozenset) -> float:
    """EWER: mean WER over utterances restricted to domain tokens in reference."""
    scores = []
    for ref, hyp in zip(refs, hyps):
        rw = [w for w in _normalize_text(ref).split() if w in domain]
        if not rw:
            continue
        sub_ref = " ".join(rw)
        sub_hyp = " ".join(w for w in _normalize_text(hyp).split() if w in domain)
        if not sub_hyp.strip():
            scores.append(1.0)
        else:
            scores.append(float(compute_wer_jiwer(sub_ref, sub_hyp)))
    if not scores:
        return float("nan")
    return float(np.mean(scores) * 100.0)


def domain_term_precision_recall_f1(ref: str, hyp: str, domain: frozenset) -> Tuple[float, float, float]:
    """Token-level precision/recall/F1 on domain vocabulary occurrences."""
    ref_toks = _normalize_text(ref).split()
    hyp_toks = set(_normalize_text(hyp).split())
    ref_dom = [t for t in ref_toks if t in domain]
    if not ref_dom:
        return (float("nan"), float("nan"), float("nan"))
    tp = sum(1 for t in ref_dom if t in hyp_toks)
    prec = tp / max(1, len([t for t in hyp_toks if t in domain]))
    rec = tp / len(ref_dom)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return (prec, rec, f1)


def aggregate_f1(refs: Sequence[str], hyps: Sequence[str], domain: frozenset) -> Tuple[float, float, float]:
    precs, recs, f1s = [], [], []
    for r, h in zip(refs, hyps):
        p, r_, f = domain_term_precision_recall_f1(r, h, domain)
        if not math.isnan(p):
            precs.append(p)
            recs.append(r_)
            f1s.append(f)
    if not f1s:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.nanmean(precs)), float(np.nanmean(recs)), float(np.nanmean(f1s)))


def sentence_error_rate(refs: Sequence[str], hyps: Sequence[str]) -> float:
    wrong = 0
    for r, h in zip(refs, hyps):
        if _normalize_text(r) != _normalize_text(h):
            wrong += 1
    return 100.0 * wrong / max(1, len(refs))


def transcribe_manifest(model: EncDecCTCModelBPE, manifest_path: str) -> Tuple[List[str], List[str]]:
    refs: List[str] = []
    paths: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            refs.append(e["text"])
            paths.append(e["audio_filepath"])
    hyps_raw = model.transcribe(paths, batch_size=CFG.BATCH_SIZE)
    hyps: List[str] = []
    for x in hyps_raw:
        if isinstance(x, str):
            hyps.append(x)
        elif hasattr(x, "text"):
            hyps.append(str(x.text))
        else:
            hyps.append(str(x))
    return refs, hyps


def evaluate_manifest_bundle(model: EncDecCTCModelBPE, manifest_path: str) -> Dict[str, Any]:
    refs, hyps = transcribe_manifest(model, manifest_path)
    wer = float(compute_wer_jiwer(refs, hyps) * 100.0)
    cer = float(compute_cer_jiwer(refs, hyps) * 100.0)
    ser = float(sentence_error_rate(refs, hyps))
    dom = current_domain_terms()
    ewer = entity_wer_from_text(refs, hyps, dom)
    p, r, f1 = aggregate_f1(refs, hyps, dom)
    return {
        "wer": wer,
        "cer": cer,
        "ser": ser,
        "ewer": ewer,
        "domain_term_precision": p,
        "domain_term_recall": r,
        "domain_term_f1": f1,
        "n_utterances": len(refs),
        "_refs": refs,
        "_hyps": hyps,
    }


def paired_bootstrap_wer_pvalue(
    refs: Sequence[str],
    hyp_a: Sequence[str],
    hyp_b: Sequence[str],
    iters: int,
    seed: int,
) -> float:
    """Approximate two-sided p-value for mean WER difference (A vs B) on paired utterances."""
    n = len(refs)
    if n < 2:
        return 1.0

    def mean_wer(hyps: Sequence[str]) -> float:
        return float(np.mean([compute_wer_jiwer(refs[i], hyps[i]) for i in range(n)]))

    obs = mean_wer(hyp_a) - mean_wer(hyp_b)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(iters):
        idx = rng.randint(0, n, size=n)
        d = np.mean([compute_wer_jiwer(refs[i], hyp_a[i]) for i in idx]) - np.mean(
            [compute_wer_jiwer(refs[i], hyp_b[i]) for i in idx]
        )
        if abs(d) >= abs(obs):
            count += 1
    return max(1.0 / (iters + 1), count / iters)


# ===========================================================================
# DATA CONFIG (NeMo)
# ===========================================================================


def build_data_config(train_manifest: str, val_manifest: str) -> DictConfig:
    nw = 0 if CFG.SMOKE_TEST else 4
    return OmegaConf.create(
        {
            "train_ds": {
                "manifest_filepath": train_manifest,
                "sample_rate": 16000,
                "batch_size": CFG.BATCH_SIZE,
                "shuffle": True,
                "num_workers": nw,
                "pin_memory": torch.cuda.is_available(),
                "trim_silence": False,
                "max_duration": 20.0,
                "min_duration": 0.5,
            },
            "validation_ds": {
                "manifest_filepath": val_manifest,
                "sample_rate": 16000,
                "batch_size": CFG.BATCH_SIZE,
                "shuffle": False,
                "num_workers": nw,
                "pin_memory": torch.cuda.is_available(),
            },
        }
    )


# ===========================================================================
# CALLBACKS
# ===========================================================================


class NemoTrainingLogger(Callback):
    """Writes per-epoch train/val metrics to CSV."""

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []
        if self.csv_path.exists():
            self.csv_path.unlink()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v.item()) if hasattr(v, "item") else float(v)
            except (TypeError, ValueError):
                metrics[k] = str(v)
        row = {"epoch": int(trainer.current_epoch), "stage": "train_end"}
        row.update(metrics)
        self.rows.append(row)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v.item()) if hasattr(v, "item") else float(v)
            except (TypeError, ValueError):
                metrics[k] = str(v)
        row = {"epoch": int(trainer.current_epoch), "stage": "val_end"}
        row.update(metrics)
        self.rows.append(row)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in self.rows for k in r.keys()}))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)


# ===========================================================================
# MODEL + RL PATCH
# ===========================================================================


def maybe_attach_lora(model: EncDecCTCModelBPE) -> None:
    """Best-effort NeMo encoder adapter (API differs across NeMo versions)."""
    if not CFG.USE_LORA:
        return
    try:
        if hasattr(model, "add_adapter"):
            cfg = OmegaConf.create(
                {
                    "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                    "in_features": 512,
                    "dim": 32,
                    "activation": "swish",
                    "norm_position": "pre",
                }
            )
            model.add_adapter(name="encoder:adapter", cfg=cfg)
            if hasattr(model, "set_enabled_adapters"):
                model.set_enabled_adapters(enabled=True)
            logger.info("Encoder adapter attached (see NeMo docs / version for exact module).")
    except Exception as e:
        logger.warning(
            "Could not attach NeMo adapter automatically (%s). Run full fine-tuning or adjust cfg for your NeMo version.",
            e,
        )


def load_model_for_sft(model_name: str) -> EncDecCTCModelBPE:
    logger.info("Loading model for SFT: %s", model_name)
    model = EncDecCTCModelBPE.from_pretrained(model_name)
    model.reward_weight = 0.0
    model._step_logs = []
    model._reward_batch_idx = 0
    model._cached_batch_reward: Optional[torch.Tensor] = None
    maybe_attach_lora(model)
    return model


def load_model_for_rl(checkpoint_path: str, reward_mode: str, reward_weight: float) -> EncDecCTCModelBPE:
    logger.info("Loading model for RL from %s (mode=%s, w=%s)", checkpoint_path, reward_mode, reward_weight)
    model = EncDecCTCModelBPE.restore_from(checkpoint_path)
    model.reward_mode = reward_mode
    model.reward_weight = reward_weight
    model._step_logs = []
    model._reward_batch_idx = 0
    model._cached_batch_reward = None
    maybe_attach_lora(model)

    original_training_step = model.training_step

    def patched_training_step(self, batch, batch_idx):
        result = original_training_step(batch, batch_idx)
        if self.reward_weight == 0.0:
            return result

        ctc_loss = result["loss"] if isinstance(result, dict) else result

        if hasattr(batch, "audio"):
            signal, signal_len = batch.audio, batch.audio_lens
            transcript, transcript_len = batch.tokens, batch.token_lens
        elif hasattr(batch, "input_signal"):
            signal, signal_len = batch.input_signal, batch.input_signal_length
            transcript, transcript_len = batch.targets, batch.target_lengths
        else:
            signal, signal_len, transcript, transcript_len = batch[:4]

        batch_sz = int(transcript.size(0))
        long_audio = bool((signal_len.max().item() > CFG.MAX_ENCODER_LEN_FOR_REWARD))

        compute_now = (batch_idx % CFG.REWARD_STEP_INTERVAL == 0) and not long_audio
        if long_audio:
            if self._cached_batch_reward is not None:
                rewards = self._cached_batch_reward.to(ctc_loss.device)
            else:
                rewards = torch.ones(batch_sz, device=ctc_loss.device, dtype=torch.float32) * 0.5
        elif not compute_now:
            if self._cached_batch_reward is not None:
                rewards = self._cached_batch_reward.to(ctc_loss.device)
            else:
                rewards = torch.ones(batch_sz, device=ctc_loss.device, dtype=torch.float32) * 0.5
        else:
            with torch.no_grad():
                log_probs, encoded_len, _ = self.forward(input_signal=signal, input_signal_length=signal_len)
                hyps_raw = self.wer.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=log_probs,
                    decoder_lengths=encoded_len,
                    return_hypotheses=False,
                )
                hyps = [
                    h.text if hasattr(h, "text") else str(h) if not isinstance(h, str) else h for h in hyps_raw
                ]
                refs = []
                for i in range(transcript.size(0)):
                    tl = transcript_len[i].item()
                    ids = transcript[i, :tl].tolist()
                    refs.append(self.tokenizer.ids_to_text(ids))

            if self.reward_mode == "mwer":
                rewards = compute_mwer_reward(hyps, refs)
            elif self.reward_mode == "wwer":
                rewards = compute_wwer_reward(hyps, refs)
            elif self.reward_mode == "llm":
                rewards = compute_llm_reward(hyps, refs)
            elif self.reward_mode == "all":
                rewards = compute_combined_reward(hyps, refs)
            else:
                rewards = compute_mwer_reward(hyps, refs)

            rewards = rewards.to(ctc_loss.device)
            if rewards.numel() == batch_sz:
                self._cached_batch_reward = rewards.detach().cpu()
            else:
                self._cached_batch_reward = None

        rewards = rewards.to(ctc_loss.device)
        if rewards.numel() != batch_sz:
            rewards = torch.ones(batch_sz, device=ctc_loss.device, dtype=torch.float32) * 0.5
        penalty = 1.0 - rewards.mean()
        total_loss = ctc_loss + self.reward_weight * penalty

        self._step_logs.append(
            {
                "step": int(batch_idx),
                "ctc_loss": float(ctc_loss.item()),
                "reward_mean": float(rewards.mean().item()),
                "penalty": float(penalty.item()),
                "total_loss": float(total_loss.item()),
                "long_audio": int(long_audio),
                "reward_skipped": int(not compute_now),
            }
        )

        if isinstance(result, dict):
            result["loss"] = total_loss
            return result
        return total_loss

    model.training_step = types.MethodType(patched_training_step, model)
    logger.info("Reward injection enabled (every %d steps, long_audio guard=%d)", CFG.REWARD_STEP_INTERVAL, CFG.MAX_ENCODER_LEN_FOR_REWARD)
    return model


# ===========================================================================
# TRAINER
# ===========================================================================


def _trainer_precision() -> str:
    return "16-mixed" if torch.cuda.is_available() else "32-true"


def compute_warmup_steps(num_samples: int, batch_size: int, num_epochs: int) -> int:
    steps_per_epoch = max(1, math.ceil(num_samples / max(1, batch_size)))
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup = max(1, min(500, int(total_steps * 0.1)))
    warmup = min(warmup, total_steps - 1)
    logger.info("  Warmup steps: %d / total %d", warmup, total_steps)
    return warmup


def create_trainer(num_epochs: int, warmup_steps: int, lr: float, csv_log: Path) -> Tuple[pl.Trainer, DictConfig]:
    cb = NemoTrainingLogger(csv_log)
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=num_epochs,
        precision=_trainer_precision(),
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=False,
        logger=False,
        callbacks=[cb],
    )
    optim = OmegaConf.create(
        {
            "name": "adamw",
            "lr": lr,
            "weight_decay": 1e-3,
            "sched": {"name": "CosineAnnealing", "warmup_steps": warmup_steps, "min_lr": 1e-6},
        }
    )
    return trainer, optim


def count_manifest_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ===========================================================================
# GCS UPLOAD
# ===========================================================================


def upload_to_gcs(local_path: str, gcs_prefix: Optional[str]) -> None:
    if CFG.SKIP_GCS or not gcs_prefix or CFG.SMOKE_TEST:
        return
    dest = gcs_prefix.rstrip("/") + "/" + Path(local_path).name
    try:
        subprocess.run(["gsutil", "-q", "cp", local_path, dest], check=True)
        logger.info("Uploaded %s -> %s", local_path, dest)
    except Exception as e:
        logger.warning("gsutil upload failed (%s). Is gcloud authenticated?", e)


def upload_dir_to_gcs(local_dir: str, gcs_prefix: Optional[str]) -> None:
    if CFG.SKIP_GCS or not gcs_prefix or CFG.SMOKE_TEST:
        return
    dest = gcs_prefix.rstrip("/") + "/" + Path(local_dir).name + "/"
    try:
        subprocess.run(["gsutil", "-q", "cp", "-r", local_dir, dest], check=True)
        logger.info("Uploaded dir %s -> %s", local_dir, dest)
    except Exception as e:
        logger.warning("gsutil recursive upload failed: %s", e)


# ===========================================================================
# STAGES
# ===========================================================================


def run_sft_stage(train_manifest: str, val_manifest: str, save_path: str, csv_path: Path) -> Tuple[EncDecCTCModelBPE, Dict[str, Any]]:
    logger.info("=" * 60 + "\nSTAGE 1: SFT\n" + "=" * 60)
    model = load_model_for_sft(CFG.NEMO_MODEL_NAME)
    data_cfg = build_data_config(train_manifest, val_manifest)
    model.setup_training_data(data_cfg.train_ds)
    model.setup_validation_data(data_cfg.validation_ds)
    n_train = count_manifest_lines(train_manifest)
    warmup = compute_warmup_steps(n_train, CFG.BATCH_SIZE, CFG.SFT_EPOCHS)
    trainer, optim = create_trainer(CFG.SFT_EPOCHS, warmup, CFG.LEARNING_RATE_SFT, csv_path)
    model.set_trainer(trainer)
    model.setup_optimization(optim)
    t0 = time.time()
    trainer.fit(model)
    train_time = time.time() - t0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_to(save_path)
    upload_to_gcs(save_path, CFG.UPLOAD_GCS_URI)
    metrics = evaluate_manifest_bundle(model, val_manifest)
    metrics["train_time_s"] = train_time
    for k in ("_refs", "_hyps"):
        metrics.pop(k, None)
    return model, metrics


def run_rl_stage(
    sft_checkpoint: str,
    train_manifest: str,
    val_manifest: str,
    save_path: str,
    csv_path: Path,
) -> Tuple[EncDecCTCModelBPE, Dict[str, Any]]:
    logger.info("=" * 60 + "\nSTAGE 2: RL (%s)\n" % CFG.REWARD_MODE + "=" * 60)
    model = load_model_for_rl(sft_checkpoint, CFG.REWARD_MODE, CFG.REWARD_WEIGHT)
    data_cfg = build_data_config(train_manifest, val_manifest)
    model.setup_training_data(data_cfg.train_ds)
    model.setup_validation_data(data_cfg.validation_ds)
    n_train = count_manifest_lines(train_manifest)
    warmup = compute_warmup_steps(n_train, CFG.BATCH_SIZE, CFG.RL_EPOCHS)
    trainer, optim = create_trainer(CFG.RL_EPOCHS, warmup, CFG.LEARNING_RATE_RL, csv_path)
    model.set_trainer(trainer)
    model.setup_optimization(optim)
    t0 = time.time()
    trainer.fit(model)
    train_time = time.time() - t0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_to(save_path)
    upload_to_gcs(save_path, CFG.UPLOAD_GCS_URI)
    metrics = evaluate_manifest_bundle(model, val_manifest)
    metrics["train_time_s"] = train_time
    traj = [float(x["reward_mean"]) for x in model._step_logs]
    metrics["reward_trajectory"] = traj
    metrics["reward_mean"] = float(np.mean(traj)) if traj else float("nan")
    metrics["reward_std"] = float(np.std(traj)) if traj else float("nan")
    for k in ("_refs", "_hyps"):
        metrics.pop(k, None)
    return model, metrics


def evaluate_zero_shot_bundle(val_manifest: str) -> Dict[str, Any]:
    logger.info("Zero-shot evaluation on validation manifest …")
    m = load_model_for_sft(CFG.NEMO_MODEL_NAME)
    out = evaluate_manifest_bundle(m, val_manifest)
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    for k in ("_refs", "_hyps"):
        out.pop(k, None)
    return out


# ===========================================================================
# PIPELINE
# ===========================================================================


def _dataset_loader_kwargs() -> dict:
    """Translate the active CFG into keyword arguments for the data module loaders."""
    if CFG.DATASET == "afrispeech_clinical":
        return dict(
            train_n=CFG.TRAIN_SAMPLES,
            val_n=CFG.VAL_SAMPLES,
            test_n=CFG.TEST_SAMPLES,
            load_test=CFG.RUN_FINAL_TEST_EVAL,
        )
    if CFG.DATASET == "librispeech":
        val_n = min(50, 2_703) if CFG.SMOKE_TEST else 2_703
        return dict(train_n=CFG.LIBRISPEECH_TRAIN_CAP, val_n=val_n)
    if CFG.DATASET == "voxpopuli":
        train_n = min(40, CFG.VOXPOPULI_TRAIN_SUBSET) if CFG.SMOKE_TEST else CFG.VOXPOPULI_TRAIN_SUBSET
        val_n = min(20, 1_750) if CFG.SMOKE_TEST else 1_750
        return dict(train_n=train_n, val_n=val_n, seed=CFG.SEED)
    return {}


def prepare_manifests() -> Dict[str, str]:
    ds, text_field = load_dataset_bundle(CFG.DATASET, **_dataset_loader_kwargs())
    # Audio is written to data/audio/<dataset_name>/ so clips from different
    # datasets never share a directory and WAV filenames cannot collide.
    audio_dir = os.path.join(_SHARED_AUDIO_DIR, CFG.DATASET)
    out: Dict[str, str] = {}
    out["train"] = build_nemo_manifest(ds["train"], CFG.DATASET, "train", audio_dir, _SHARED_MANIFEST_DIR, text_field)
    out["val"] = build_nemo_manifest(ds["validation"], CFG.DATASET, "val", audio_dir, _SHARED_MANIFEST_DIR, text_field)
    if "test" in ds:
        out["test"] = build_nemo_manifest(ds["test"], CFG.DATASET, "test", audio_dir, _SHARED_MANIFEST_DIR, text_field)
    return out


def prepare_librispeech_eval_manifest() -> str:
    """Forgetting-eval manifest: LibriSpeech validation written to the shared data/ dir."""
    val_n = 200 if CFG.SMOKE_TEST else 2_703
    eval_ds, text_field = load_librispeech_eval(val_n=val_n)
    audio_dir = os.path.join(_SHARED_AUDIO_DIR, "librispeech")
    return build_nemo_manifest(
        eval_ds, "librispeech", "forgetting_eval", audio_dir, _SHARED_MANIFEST_DIR, text_field
    )


def catastrophic_forgetting_eval(model: EncDecCTCModelBPE, libri_manifest: str) -> Dict[str, Any]:
    m = evaluate_manifest_bundle(model, libri_manifest)
    for k in ("_refs", "_hyps"):
        m.pop(k, None)
    return m


def save_results_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote results JSON: %s", path)
    upload_to_gcs(str(path), CFG.UPLOAD_GCS_URI)


def run_full_pipeline() -> Dict[str, Any]:
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    test_reward_functions()

    manifests = prepare_manifests()
    train_m, val_m = manifests["train"], manifests["val"]
    test_m = manifests.get("test")

    run_id = f"{CFG.DATASET}_seed{CFG.SEED}_{int(time.time())}"
    cfg_dump = asdict(CFG)
    for k, v in list(cfg_dump.items()):
        if isinstance(v, frozenset):
            cfg_dump[k] = sorted(v)
        if k.startswith("_"):
            cfg_dump.pop(k, None)
    results: Dict[str, Any] = {"run_id": run_id, "config": cfg_dump}

    if CFG.RUN_ZERO_SHOT:
        results["zero_shot_val"] = evaluate_zero_shot_bundle(val_m)

    sft_ckpt = os.path.join(CFG.CHECKPOINT_DIR, "sft_model.nemo")
    sft_model, sft_metrics = run_sft_stage(
        train_m,
        val_m,
        sft_ckpt,
        Path(CFG.RESULTS_DIR) / f"{run_id}_sft_epoch_metrics.csv",
    )
    results["sft"] = sft_metrics
    results["sft_checkpoint"] = sft_ckpt

    libri_manifest: Optional[str] = None
    if CFG.RUN_LIBRISPEECH_FORGETTING:
        libri_manifest = prepare_librispeech_eval_manifest()
        results["librispeech_after_sft"] = catastrophic_forgetting_eval(sft_model, libri_manifest)

    rl_ckpt = os.path.join(CFG.CHECKPOINT_DIR, "rl_model.nemo")
    rl_model, rl_metrics = run_rl_stage(
        sft_ckpt,
        train_m,
        val_m,
        rl_ckpt,
        Path(CFG.RESULTS_DIR) / f"{run_id}_rl_epoch_metrics.csv",
    )
    results["rl"] = rl_metrics
    results["rl_checkpoint"] = rl_ckpt

    if CFG.RUN_LIBRISPEECH_FORGETTING and libri_manifest:
        results["librispeech_after_rl"] = catastrophic_forgetting_eval(rl_model, libri_manifest)

    # Paired bootstrap on validation (SFT vs RL) using stored re-eval
    sft_eval = evaluate_manifest_bundle(sft_model, val_m)
    rl_eval = evaluate_manifest_bundle(rl_model, val_m)
    if sft_eval.get("_refs") and rl_eval.get("_hyps"):
        results["paired_bootstrap_pval_sft_vs_rl_wer"] = paired_bootstrap_wer_pvalue(
            sft_eval["_refs"],
            sft_eval["_hyps"],
            rl_eval["_hyps"],
            CFG.BOOTSTRAP_ITERS,
            CFG.SEED,
        )

    if test_m and CFG.RUN_FINAL_TEST_EVAL:
        results["test_sft"] = evaluate_manifest_bundle(sft_model, test_m)
        results["test_rl"] = evaluate_manifest_bundle(rl_model, test_m)
        for section in ("test_sft", "test_rl"):
            for k in ("_refs", "_hyps"):
                results[section].pop(k, None)

    for section in list(results.keys()):
        if isinstance(results[section], dict):
            for k in ("_refs", "_hyps"):
                results[section].pop(k, None)

    out_path = Path(CFG.RESULTS_DIR) / f"{run_id}_results.json"
    save_results_json(results, out_path)
    upload_dir_to_gcs(_SHARED_MANIFEST_DIR, CFG.UPLOAD_GCS_URI)

    logger.info("Done. Key val WER: sft=%.2f rl=%.2f", sft_metrics["wer"], rl_metrics["wer"])
    return results


def run_sft_only() -> Dict[str, Any]:
    manifests = prepare_manifests()
    run_id = f"{CFG.DATASET}_seed{CFG.SEED}_sft_{int(time.time())}"
    _, sft_metrics = run_sft_stage(
        manifests["train"],
        manifests["val"],
        os.path.join(CFG.CHECKPOINT_DIR, "sft_model.nemo"),
        Path(CFG.RESULTS_DIR) / f"{run_id}_sft_epoch_metrics.csv",
    )
    save_results_json({"run_id": run_id, "sft": sft_metrics}, Path(CFG.RESULTS_DIR) / f"{run_id}_results.json")
    return sft_metrics


def run_rl_only(sft_checkpoint: str) -> Dict[str, Any]:
    manifests = prepare_manifests()
    run_id = f"{CFG.DATASET}_seed{CFG.SEED}_rl_{int(time.time())}"
    _, rl_metrics = run_rl_stage(
        sft_checkpoint,
        manifests["train"],
        manifests["val"],
        os.path.join(CFG.CHECKPOINT_DIR, "rl_model.nemo"),
        Path(CFG.RESULTS_DIR) / f"{run_id}_rl_epoch_metrics.csv",
    )
    save_results_json({"run_id": run_id, "rl": rl_metrics}, Path(CFG.RESULTS_DIR) / f"{run_id}_results.json")
    return rl_metrics


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo STT SFT + RL-style reward training")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--stage", choices=["sft", "rl", "both"], default="both")
    p.add_argument("--sft_checkpoint", type=str, default=None)
    p.add_argument("--dataset", choices=["afrispeech_clinical", "librispeech", "voxpopuli"], default="afrispeech_clinical")
    p.add_argument("--train_samples", type=int, default=None, help="Cap AfriSpeech clinical train count (None=all)")
    p.add_argument("--val_samples", type=int, default=None, help="Cap AfriSpeech clinical val count (None=all)")
    p.add_argument("--test_samples", type=int, default=None, help="Cap AfriSpeech clinical test count (None=all)")
    p.add_argument("--voxpopuli_train_subset", type=int, default=None)
    p.add_argument("--reward_mode", type=str, default=None)
    p.add_argument("--reward_weight", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--upload_gcs", type=str, default=None, help="gs://bucket/prefix — checkpoints + results")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--mock_llm", action="store_true", help="Force mock LLM reward")
    p.add_argument("--real_llm", action="store_true", help="Use Gemini for llm reward (needs GEMINI_API_KEY)")
    p.add_argument("--skip_zero_shot", action="store_true")
    p.add_argument("--skip_librispeech_forgetting", action="store_true")
    p.add_argument("--skip_test_eval", action="store_true")
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Train/eval batch size (use 8 on V100 16GB for stt_en_conformer_ctc_medium if OOM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        apply_smoke_test_overrides()
    CFG.DATASET = args.dataset
    CFG.SEED = args.seed
    if args.train_samples is not None:
        CFG.TRAIN_SAMPLES = args.train_samples
    if args.val_samples is not None:
        CFG.VAL_SAMPLES = args.val_samples
    if args.test_samples is not None:
        CFG.TEST_SAMPLES = args.test_samples
    if args.voxpopuli_train_subset is not None:
        CFG.VOXPOPULI_TRAIN_SUBSET = args.voxpopuli_train_subset
    if args.reward_mode:
        CFG.REWARD_MODE = args.reward_mode
    if args.reward_weight is not None:
        CFG.REWARD_WEIGHT = args.reward_weight
    if args.upload_gcs:
        CFG.UPLOAD_GCS_URI = args.upload_gcs.rstrip("/")
    CFG.USE_LORA = args.use_lora
    if args.mock_llm:
        CFG.USE_MOCK_LLM = True
    if args.real_llm:
        CFG.USE_MOCK_LLM = False
    if args.skip_zero_shot:
        CFG.RUN_ZERO_SHOT = False
    if args.skip_librispeech_forgetting:
        CFG.RUN_LIBRISPEECH_FORGETTING = False
    if args.skip_test_eval:
        CFG.RUN_FINAL_TEST_EVAL = False
    if args.batch_size is not None and not args.smoke_test:
        CFG.BATCH_SIZE = args.batch_size

    set_seed(CFG.SEED)

    if not args.smoke_test:
        # Defaults for “full” AfriSpeech run
        if CFG.DATASET == "afrispeech_clinical" and args.train_samples is None:
            CFG.TRAIN_SAMPLES = None
        if CFG.DATASET == "afrispeech_clinical" and args.val_samples is None:
            CFG.VAL_SAMPLES = None
        if CFG.DATASET == "afrispeech_clinical" and args.test_samples is None:
            CFG.TEST_SAMPLES = None

    if args.stage == "sft":
        run_sft_only()
    elif args.stage == "rl":
        if not args.sft_checkpoint:
            raise SystemExit("--sft_checkpoint required for --stage rl")
        run_rl_only(args.sft_checkpoint)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
