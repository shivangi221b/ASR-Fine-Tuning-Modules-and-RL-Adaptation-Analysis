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
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jiwer import cer as compute_cer_jiwer
from jiwer import wer as compute_wer_jiwer
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Repo root and shared data paths
# ---------------------------------------------------------------------------
def _detect_repo_root(start: Path) -> Path:
    """
    Find the repo root robustly.

    This script is usually located at nemo/gcp_scripts/, but on some VMs it may
    be copied to the repo root. We search upward for a directory that contains
    the repo's `data/` folder.
    """
    p = start.resolve()
    for cand in [p] + list(p.parents)[:8]:
        if (cand / "data").is_dir():
            return cand
    # Fallback to previous assumption (keeps behavior stable if `data/` is absent).
    return p.parents[2] if len(p.parents) >= 3 else p.parent


_REPO_ROOT = _detect_repo_root(Path(__file__).parent)
_SHARED_MANIFEST_DIR = str(_REPO_ROOT / "data" / "manifests")
_SHARED_AUDIO_DIR = str(_REPO_ROOT / "data" / "audio")

sys.path.insert(0, str(_REPO_ROOT))
from data import (  # noqa: E402
    build_nemo_manifest,
    load_dataset_bundle,
    load_librispeech_eval,
    prepare_afrispeech_clinical_manifests_streaming,
)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch import LightningModule
from nemo.collections.asr.models import EncDecCTCModelBPE


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
    # Reward compute is expensive because it does forward+decode+string scoring.
    # Guard it for long utterances. NOTE: batch signal lengths are typically in
    # *audio samples*, not encoder frames, so we guard on seconds (sample-based)
    # to avoid accidentally skipping reward on every batch.
    MAX_AUDIO_SECONDS_FOR_REWARD: float = 25.0
    SAMPLE_RATE: int = 16000

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
    DEBUG_REWARD: bool = False
    DEBUG_LOG_EVERY_N_STEPS: int = 200
    DEBUG_SAMPLE_DUMP: bool = False
    DEBUG_SAMPLE_EVERY_N_STEPS: int = 200
    DEBUG_SAMPLE_COUNT: int = 10
    EMPTY_HYP_WARN_FRAC: float = 0.5
    NORMALIZE_TEXT: bool = False
    TOKENIZER_UNK_GUARD: bool = True
    TOKENIZER_UNK_WARN_FRAC: float = 0.2
    # Default to fp32 for stability (esp. CTC); override with --force_fp32 false not supported,
    # but you can implement an --amp flag later if you want 16-mixed throughput.
    FORCE_FP32: bool = True
    GRAD_CLIP_VAL: float = 1.0  # enable by default (helps CTC stability)

    # Stage-2 objective: make reward actually affect gradients
    # - "reweight_ctc": compute per-sample CTC loss and reweight by (1 + w*(1-reward))
    # - "add_penalty": old behavior (adds constant penalty; does not change gradients)
    RL_OBJECTIVE: str = "reweight_ctc"


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
    # In practice, AfriSpeech clinical utterances can exceed 8s, and the reward
    # debug path would be skipped entirely. Use a higher threshold for smoke tests
    # so we actually exercise the reward compute path.
    CFG.MAX_AUDIO_SECONDS_FOR_REWARD = 30.0
    CFG.DEBUG_REWARD = True
    CFG.DEBUG_LOG_EVERY_N_STEPS = 5
    CFG.DEBUG_SAMPLE_DUMP = True
    CFG.DEBUG_SAMPLE_EVERY_N_STEPS = 5
    CFG.DEBUG_SAMPLE_COUNT = 10
    CFG.EMPTY_HYP_WARN_FRAC = 0.2
    CFG.NORMALIZE_TEXT = True
    CFG.TOKENIZER_UNK_GUARD = True
    CFG.TOKENIZER_UNK_WARN_FRAC = 0.1
    # Smoke tests are for debugging logic, not peak throughput.
    # AMP can introduce NaNs quickly on tiny datasets; force fp32 here.
    CFG.FORCE_FP32 = True
    CFG.GRAD_CLIP_VAL = 1.0
    CFG.RL_OBJECTIVE = "reweight_ctc"


# ===========================================================================
# DEBUG SAMPLE DUMPS
# ===========================================================================

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")


def normalize_text_for_ctc(s: str) -> str:
    """
    Conservative text normalization for CTC+BPE models.

    Goal: reduce tokenizer UNK pressure from punctuation/odd glyphs in AfriSpeech
    transcripts during small-sample debugging.
    """
    s = s.lower().strip()
    s = s.replace("\u2047", " ")  # NeMo logs show '⁇' glyph; treat as junk if present
    s = _NON_ALNUM_RE.sub(" ", s)
    return " ".join(s.split())


def maybe_write_normalized_manifest(src_manifest: str, dst_manifest: str) -> str:
    if not CFG.NORMALIZE_TEXT:
        return src_manifest
    Path(dst_manifest).parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    with open(src_manifest, "r", encoding="utf-8") as fin, open(dst_manifest, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            txt = str(e.get("text", ""))
            e["text"] = normalize_text_for_ctc(txt)
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
            n_out += 1
    logger.info("Wrote normalized manifest: %s (rows=%d -> %d)", dst_manifest, n_in, n_out)
    return dst_manifest


def tokenizer_unk_stats(manifest_path: str, tokenizer: Any, sample_n: int = 200, seed: int = 0) -> Dict[str, Any]:
    """
    Approximate how often target text maps to tokenizer UNK.
    High UNK pressure can cause the model to learn to emit only UNK ('⁇').
    """
    rows = load_manifest_examples(manifest_path, n=sample_n, seed=seed)
    if not rows:
        return {"n": 0}
    # Try to locate unk id
    unk_id = None
    for attr in ("unk_id", "unk"):
        if hasattr(tokenizer, attr):
            try:
                unk_id = int(getattr(tokenizer, attr))
                break
            except Exception:
                pass
    # NeMo SentencePieceTokenizer exposes `tokenizer.unk_id` via inner model sometimes
    if unk_id is None and hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "unk_id"):
        try:
            unk_id = int(tokenizer.tokenizer.unk_id())
        except Exception:
            unk_id = None

    unk_fracs: List[float] = []
    lens: List[int] = []
    for r in rows:
        t = str(r.get("text", ""))
        try:
            ids = tokenizer.text_to_ids(t)
        except Exception:
            continue
        ids = list(ids) if isinstance(ids, (list, tuple)) else list(getattr(ids, "tolist", lambda: [])())
        if not ids:
            continue
        lens.append(len(ids))
        if unk_id is None:
            continue
        unk_fracs.append(sum(1 for i in ids if int(i) == unk_id) / len(ids))
    out: Dict[str, Any] = {
        "n": len(lens),
        "mean_len_ids": float(np.mean(lens)) if lens else float("nan"),
        "unk_id": unk_id,
    }
    if unk_fracs:
        out["mean_unk_frac"] = float(np.mean(unk_fracs))
        out["p95_unk_frac"] = float(np.percentile(np.array(unk_fracs), 95))
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def load_manifest_examples(manifest_path: str, n: int, seed: int) -> List[Dict[str, str]]:
    """Load N (path,text) examples from a NeMo manifest JSONL."""
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = str(e.get("audio_filepath", "")).strip()
            t = str(e.get("text", "")).strip()
            if p:
                rows.append({"audio_filepath": p, "text": t})
    if not rows:
        return []
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: max(1, n)]


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize_empty_hyps(hyps: Sequence[str]) -> Tuple[float, float]:
    """Return (empty_fraction, mean_len_chars)."""
    if not hyps:
        return (1.0, 0.0)
    empty = sum(1 for h in hyps if not str(h).strip())
    mean_len = float(np.mean([len(str(h).strip()) for h in hyps])) if hyps else 0.0
    return (empty / max(1, len(hyps)), mean_len)


def summarize_degenerate_hyps(hyps: Sequence[str]) -> Tuple[float, float]:
    """
    Return (degenerate_fraction, mean_len_chars).

    Treat hypotheses that are empty OR effectively only the SentencePiece/NeMo
    unknown token glyph ('⁇') as degenerate. This is the failure mode we saw in
    debug dumps (hypothesis_text == ' ⁇ ' repeated).
    """
    if not hyps:
        return (1.0, 0.0)
    cleaned = [str(h).strip() for h in hyps]
    deg = 0
    for s in cleaned:
        if not s:
            deg += 1
            continue
        # collapse whitespace and check if only unknown glyph(s)
        ss = " ".join(s.split())
        if ss.replace("⁇", "").strip() == "":
            deg += 1
    mean_len = float(np.mean([len(s) for s in cleaned])) if cleaned else 0.0
    return (deg / max(1, len(cleaned)), mean_len)


class SampleTranscriptionDumper(Callback):
    """Periodically transcribe a fixed small set of files and dump ref/hyp pairs."""

    def __init__(
        self,
        run_dir: Path,
        stage: str,
        val_manifest: str,
        dump_every_n_steps: int,
        n_examples: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.stage = stage
        self.dump_every_n_steps = max(1, int(dump_every_n_steps))
        self.examples = load_manifest_examples(val_manifest, n=n_examples, seed=seed)
        self.path = self.run_dir / "debug" / f"debug_samples_{stage}.jsonl"

    def _dump(self, trainer: pl.Trainer, pl_module: LightningModule, reason: str) -> None:
        if not self.examples:
            return
        model = pl_module  # NeMo model is also a LightningModule
        paths = [e["audio_filepath"] for e in self.examples]
        refs = [e.get("text", "") for e in self.examples]
        try:
            # Ensure deterministic eval-time behavior even when called mid-training.
            was_training = bool(getattr(model, "training", False))
            model.eval()
            with torch.no_grad():
                hyps_raw = model.transcribe(paths, batch_size=min(CFG.BATCH_SIZE, len(paths)))  # type: ignore[attr-defined]
            if was_training:
                model.train()
        except Exception as e:
            append_jsonl(
                self.path,
                [
                    {
                        "ts": time.time(),
                        "stage": self.stage,
                        "reason": reason,
                        "epoch": int(getattr(trainer, "current_epoch", -1)),
                        "global_step": int(getattr(trainer, "global_step", -1)),
                        "error": f"{type(e).__name__}: {e}",
                    }
                ],
            )
            return

        hyps: List[str] = []
        for x in hyps_raw:
            if isinstance(x, str):
                hyps.append(x)
            elif hasattr(x, "text"):
                hyps.append(str(x.text))
            else:
                hyps.append(str(x))

        empty_frac, _ = summarize_empty_hyps(hyps)
        deg_frac, mean_len = summarize_degenerate_hyps(hyps)
        header = {
            "ts": time.time(),
            "stage": self.stage,
            "reason": reason,
            "epoch": int(getattr(trainer, "current_epoch", -1)),
            "global_step": int(getattr(trainer, "global_step", -1)),
            "reward_mode": getattr(model, "reward_mode", None),
            "reward_weight": _safe_float(getattr(model, "reward_weight", None)),
            "empty_hyp_frac": empty_frac,
            "degenerate_hyp_frac": deg_frac,
            "mean_hyp_len_chars": mean_len,
        }
        records: List[Dict[str, Any]] = [header]
        for p, r, h in zip(paths, refs, hyps):
            records.append(
                {
                    "stage": self.stage,
                    "audio_filepath": p,
                    "reference_text": r,
                    "hypothesis_text": h,
                }
            )
        append_jsonl(self.path, records)

        if deg_frac >= float(CFG.EMPTY_HYP_WARN_FRAC):
            logger.warning(
                "[debug-samples] High degenerate hypothesis rate: %.1f%% (stage=%s, step=%s). Wrote %s",
                100.0 * deg_frac,
                self.stage,
                int(getattr(trainer, "global_step", -1)),
                str(self.path),
            )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        if CFG.DEBUG_SAMPLE_DUMP:
            self._dump(trainer, pl_module, reason="val_epoch_end")

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not CFG.DEBUG_SAMPLE_DUMP:
            return
        step = int(getattr(trainer, "global_step", 0))
        if step > 0 and (step % self.dump_every_n_steps == 0):
            self._dump(trainer, pl_module, reason=f"train_step_{step}")


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
    # Ensure evaluation transcribe runs in eval/no_grad even if caller left the
    # model in train mode after fitting.
    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        hyps_raw = model.transcribe(paths, batch_size=CFG.BATCH_SIZE)
    if was_training:
        model.train()
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
        self._fieldnames: List[str] = []
        if self.csv_path.exists():
            self.csv_path.unlink()

    def _coerce_metrics(self, trainer: pl.Trainer) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v.item()) if hasattr(v, "item") else float(v)
            except (TypeError, ValueError):
                metrics[k] = str(v)
        return metrics

    def _ensure_header_and_write(self) -> None:
        # Callback metrics keys can vary over time; keep CSV parseable by
        # rewriting the entire file when new columns appear.
        fieldnames = sorted({k for r in self.rows for k in r.keys()})
        if fieldnames != self._fieldnames:
            self._fieldnames = fieldnames
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames)
                w.writeheader()
                for r in self.rows:
                    w.writerow(r)
            return

        # Fast path: append last row.
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            w.writerow(self.rows[-1])

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        metrics = self._coerce_metrics(trainer)
        row = {"epoch": int(trainer.current_epoch), "stage": "train_end"}
        row.update(metrics)
        self.rows.append(row)
        self._ensure_header_and_write()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        metrics = self._coerce_metrics(trainer)
        row = {"epoch": int(trainer.current_epoch), "stage": "val_end"}
        row.update(metrics)
        self.rows.append(row)
        self._ensure_header_and_write()


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
    # For tiny smoke tests, SpecAugment can sometimes destabilize/obscure debugging.
    # Disable it to make “blank hypothesis collapse” easier to diagnose.
    if CFG.SMOKE_TEST and hasattr(model, "spec_augmentation"):
        try:
            model.spec_augmentation = None
        except Exception:
            pass
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
    if CFG.SMOKE_TEST and hasattr(model, "spec_augmentation"):
        try:
            model.spec_augmentation = None
        except Exception:
            pass
    maybe_attach_lora(model)

    original_training_step = model.training_step

    def patched_training_step(self, batch, batch_idx):
        # If reward is disabled, fall back entirely.
        if self.reward_weight == 0.0:
            return original_training_step(batch, batch_idx)

        if hasattr(batch, "audio"):
            signal, signal_len = batch.audio, batch.audio_lens
            transcript, transcript_len = batch.tokens, batch.token_lens
        elif hasattr(batch, "input_signal"):
            signal, signal_len = batch.input_signal, batch.input_signal_length
            transcript, transcript_len = batch.targets, batch.target_lengths
        else:
            signal, signal_len, transcript, transcript_len = batch[:4]

        batch_sz = int(transcript.size(0))
        # In NeMo batches, signal_len is usually in audio samples (not frames).
        # Guard based on duration seconds to avoid skipping reward always.
        max_samp = float(signal_len.max().item())
        max_sec = max_samp / float(CFG.SAMPLE_RATE)
        long_audio = bool(max_sec > float(CFG.MAX_AUDIO_SECONDS_FOR_REWARD))

        compute_now = (batch_idx % CFG.REWARD_STEP_INTERVAL == 0) and not long_audio
        if CFG.DEBUG_REWARD and (batch_idx < 5 or batch_idx % max(1, CFG.DEBUG_LOG_EVERY_N_STEPS) == 0):
            logger.info(
                "[reward-debug] step=%d bs=%d max_len_samples=%.0f max_len_sec=%.2f compute_now=%s long_audio=%s cached=%s",
                int(batch_idx),
                batch_sz,
                max_samp,
                max_sec,
                bool(compute_now),
                bool(long_audio),
                self._cached_batch_reward is not None,
            )
        if long_audio:
            if self._cached_batch_reward is not None:
                rewards = self._cached_batch_reward.to(ctc_loss.device)
            else:
                rewards = torch.ones(batch_sz, device=ctc_loss.device, dtype=torch.float32) * 0.5
        elif not compute_now:
            if self._cached_batch_reward is not None:
                rewards = self._cached_batch_reward
            else:
                rewards = torch.ones(batch_sz, dtype=torch.float32) * 0.5
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
                if CFG.DEBUG_REWARD and (batch_idx < 5 or batch_idx % max(1, CFG.DEBUG_LOG_EVERY_N_STEPS) == 0):
                    empty_h = sum(1 for x in hyps if not str(x).strip())
                    empty_r = sum(1 for x in refs if not str(x).strip())
                    logger.info(
                        "[reward-debug] step=%d encoded_len_max=%d empty_hyps=%d/%d empty_refs=%d/%d",
                        int(batch_idx),
                        int(encoded_len.max().item()) if hasattr(encoded_len, "max") else -1,
                        int(empty_h),
                        int(len(hyps)),
                        int(empty_r),
                        int(len(refs)),
                    )

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

            rewards = rewards.detach().cpu()
            if rewards.numel() == batch_sz:
                self._cached_batch_reward = rewards
            else:
                self._cached_batch_reward = None

        # Move reward vector to device for weighting
        rewards = rewards.to(signal.device)
        if rewards.numel() != batch_sz:
            rewards = torch.ones(batch_sz, device=signal.device, dtype=torch.float32) * 0.5

        # Compute per-sample CTC loss with gradient.
        log_probs_g, encoded_len_g, _ = self.forward(input_signal=signal, input_signal_length=signal_len)
        lp = log_probs_g.transpose(0, 1)  # [T,B,V]
        input_lengths = encoded_len_g.to(torch.int32)
        target_lengths = transcript_len.to(torch.int32)
        chunks = []
        for i in range(batch_sz):
            tl = int(target_lengths[i].item())
            if tl > 0:
                chunks.append(transcript[i, :tl].to(torch.int32))
        targets_flat = torch.cat(chunks) if chunks else torch.zeros((0,), device=lp.device, dtype=torch.int32)
        blank = 0
        for attr in ("blank_idx", "blank_id"):
            if hasattr(self.decoder, attr):
                try:
                    blank = int(getattr(self.decoder, attr))
                    break
                except Exception:
                    pass
        ctc_per = F.ctc_loss(
            lp,
            targets_flat,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=blank,
            reduction="none",
            zero_infinity=True,
        )

        # RL objective
        if CFG.RL_OBJECTIVE == "add_penalty":
            penalty = 1.0 - rewards.mean()
            total_loss = ctc_per.mean() + float(self.reward_weight) * penalty
        else:
            # Upweight low-reward samples -> changes gradient magnitudes.
            weights = 1.0 + float(self.reward_weight) * (1.0 - rewards.detach())
            total_loss = (ctc_per * weights).mean()
            penalty = 1.0 - rewards.mean()

        self._step_logs.append(
            {
                "step": int(batch_idx),
                "ctc_loss": float(ctc_per.mean().item()),
                "reward_mean": float(rewards.mean().item()),
                "penalty": float(penalty.item()),
                "total_loss": float(total_loss.item()),
                "long_audio": int(long_audio),
                "reward_skipped": int(not compute_now),
            }
        )

        return {"loss": total_loss}

    model.training_step = types.MethodType(patched_training_step, model)
    logger.info(
        "Reward injection enabled (every %d steps, long_audio guard: > %.1fs at %d Hz)",
        CFG.REWARD_STEP_INTERVAL,
        float(CFG.MAX_AUDIO_SECONDS_FOR_REWARD),
        int(CFG.SAMPLE_RATE),
    )
    return model


# ===========================================================================
# TRAINER
# ===========================================================================


def _trainer_precision() -> str:
    if CFG.FORCE_FP32:
        return "32-true"
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


def create_trainer_with_artifacts(
    *,
    num_epochs: int,
    warmup_steps: int,
    lr: float,
    csv_log: Path,
    run_dir: Path,
    stage: str,
    val_manifest: str,
) -> Tuple[pl.Trainer, DictConfig]:
    """Trainer factory that enables checkpoints + optional debug dumps."""
    cb_metrics = NemoTrainingLogger(csv_log)
    callbacks: List[Callback] = [cb_metrics]

    # Checkpointing (resume-able)
    ckpt_dir = run_dir / "checkpoints" / stage
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch}-{step}",
            save_last=True,
            every_n_epochs=1,
        )
    )

    # Debug sample dumps (text-only)
    if CFG.DEBUG_SAMPLE_DUMP:
        callbacks.append(
            SampleTranscriptionDumper(
                run_dir=run_dir,
                stage=stage,
                val_manifest=val_manifest,
                dump_every_n_steps=CFG.DEBUG_SAMPLE_EVERY_N_STEPS,
                n_examples=CFG.DEBUG_SAMPLE_COUNT,
                seed=CFG.SEED,
            )
        )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=num_epochs,
        precision=_trainer_precision(),
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=True,
        gradient_clip_val=float(CFG.GRAD_CLIP_VAL) if CFG.GRAD_CLIP_VAL else 0.0,
        logger=False,
        callbacks=callbacks,
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


def run_sft_stage(
    train_manifest: str,
    val_manifest: str,
    save_path: str,
    csv_path: Path,
    *,
    run_dir: Path,
    resume_ckpt: Optional[str] = None,
) -> Tuple[EncDecCTCModelBPE, Dict[str, Any]]:
    logger.info("=" * 60 + "\nSTAGE 1: SFT\n" + "=" * 60)
    model = load_model_for_sft(CFG.NEMO_MODEL_NAME)
    data_cfg = build_data_config(train_manifest, val_manifest)
    model.setup_training_data(data_cfg.train_ds)
    model.setup_validation_data(data_cfg.validation_ds)
    # Tokenizer UNK diagnostics (common cause of '⁇' collapse on small data).
    try:
        stats_train = tokenizer_unk_stats(train_manifest, model.tokenizer, sample_n=200, seed=CFG.SEED)
        stats_val = tokenizer_unk_stats(val_manifest, model.tokenizer, sample_n=200, seed=CFG.SEED + 1)
        logger.info("[tokenizer] train unk stats: %s", stats_train)
        logger.info("[tokenizer] val unk stats: %s", stats_val)
        if CFG.TOKENIZER_UNK_GUARD and stats_train.get("mean_unk_frac") is not None:
            if float(stats_train["mean_unk_frac"]) >= float(CFG.TOKENIZER_UNK_WARN_FRAC):
                logger.warning(
                    "[tokenizer] High mean UNK fraction in training targets (%.3f). "
                    "This often causes models to learn to emit only UNK ('⁇'). "
                    "Consider enabling --normalize_text (or improving normalization).",
                    float(stats_train["mean_unk_frac"]),
                )
    except Exception as e:
        logger.warning("[tokenizer] UNK stats failed: %s", e)
    n_train = count_manifest_lines(train_manifest)
    warmup = compute_warmup_steps(n_train, CFG.BATCH_SIZE, CFG.SFT_EPOCHS)
    trainer, optim = create_trainer_with_artifacts(
        num_epochs=CFG.SFT_EPOCHS,
        warmup_steps=warmup,
        lr=CFG.LEARNING_RATE_SFT,
        csv_log=csv_path,
        run_dir=run_dir,
        stage="sft",
        val_manifest=val_manifest,
    )
    model.set_trainer(trainer)
    model.setup_optimization(optim)
    t0 = time.time()
    trainer.fit(model, ckpt_path=resume_ckpt)
    train_time = time.time() - t0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_to(save_path)
    # Also save a stage-local copy next to checkpoints/debug artifacts.
    try:
        stage_local = run_dir / "exports" / "sft_model.nemo"
        stage_local.parent.mkdir(parents=True, exist_ok=True)
        model.save_to(str(stage_local))
    except Exception as e:
        logger.warning("Could not save stage-local SFT .nemo: %s", e)
    upload_to_gcs(save_path, CFG.UPLOAD_GCS_URI)
    metrics = evaluate_manifest_bundle(model, val_manifest)
    # Sanity guard: empty/degenerate hypotheses often manifest as WER=100 everywhere.
    if metrics.get("_hyps"):
        empty_frac, _ = summarize_empty_hyps(metrics["_hyps"])
        deg_frac, mean_len = summarize_degenerate_hyps(metrics["_hyps"])
        metrics["_empty_hyp_frac"] = empty_frac
        metrics["_degenerate_hyp_frac"] = deg_frac
        metrics["_mean_hyp_len_chars"] = mean_len
        if deg_frac >= float(CFG.EMPTY_HYP_WARN_FRAC):
            logger.warning(
                "[sanity] High degenerate hypothesis rate after SFT eval: %.1f%% (mean_len=%.1f chars).",
                100.0 * deg_frac,
                mean_len,
            )
            # Force an immediate debug dump for inspection.
            if CFG.DEBUG_SAMPLE_DUMP:
                try:
                    SampleTranscriptionDumper(
                        run_dir=run_dir,
                        stage="sft",
                        val_manifest=val_manifest,
                        dump_every_n_steps=CFG.DEBUG_SAMPLE_EVERY_N_STEPS,
                        n_examples=CFG.DEBUG_SAMPLE_COUNT,
                        seed=CFG.SEED,
                    )._dump(trainer, model, reason="forced_after_sft_eval_empty_hyps")
                except Exception as e:
                    logger.warning("[sanity] Failed to force SFT debug dump: %s", e)
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
    *,
    run_dir: Path,
    resume_ckpt: Optional[str] = None,
) -> Tuple[EncDecCTCModelBPE, Dict[str, Any]]:
    logger.info("=" * 60 + "\nSTAGE 2: RL (%s)\n" % CFG.REWARD_MODE + "=" * 60)
    model = load_model_for_rl(sft_checkpoint, CFG.REWARD_MODE, CFG.REWARD_WEIGHT)
    data_cfg = build_data_config(train_manifest, val_manifest)
    model.setup_training_data(data_cfg.train_ds)
    model.setup_validation_data(data_cfg.validation_ds)
    n_train = count_manifest_lines(train_manifest)
    warmup = compute_warmup_steps(n_train, CFG.BATCH_SIZE, CFG.RL_EPOCHS)
    trainer, optim = create_trainer_with_artifacts(
        num_epochs=CFG.RL_EPOCHS,
        warmup_steps=warmup,
        lr=CFG.LEARNING_RATE_RL,
        csv_log=csv_path,
        run_dir=run_dir,
        stage="rl",
        val_manifest=val_manifest,
    )
    model.set_trainer(trainer)
    model.setup_optimization(optim)
    t0 = time.time()
    trainer.fit(model, ckpt_path=resume_ckpt)
    train_time = time.time() - t0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_to(save_path)
    try:
        stage_local = run_dir / "exports" / "rl_model.nemo"
        stage_local.parent.mkdir(parents=True, exist_ok=True)
        model.save_to(str(stage_local))
    except Exception as e:
        logger.warning("Could not save stage-local RL .nemo: %s", e)
    upload_to_gcs(save_path, CFG.UPLOAD_GCS_URI)
    metrics = evaluate_manifest_bundle(model, val_manifest)
    if metrics.get("_hyps"):
        empty_frac, _ = summarize_empty_hyps(metrics["_hyps"])
        deg_frac, mean_len = summarize_degenerate_hyps(metrics["_hyps"])
        metrics["_empty_hyp_frac"] = empty_frac
        metrics["_degenerate_hyp_frac"] = deg_frac
        metrics["_mean_hyp_len_chars"] = mean_len
        if deg_frac >= float(CFG.EMPTY_HYP_WARN_FRAC):
            logger.warning(
                "[sanity] High degenerate hypothesis rate after RL eval: %.1f%% (mean_len=%.1f chars).",
                100.0 * deg_frac,
                mean_len,
            )
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
    if CFG.DATASET == "afrispeech_clinical":
        out = prepare_afrispeech_clinical_manifests_streaming(
            train_n=CFG.TRAIN_SAMPLES,
            val_n=CFG.VAL_SAMPLES,
            test_n=CFG.TEST_SAMPLES,
            load_test=CFG.RUN_FINAL_TEST_EVAL,
            dataset_name=CFG.DATASET,
            audio_base_dir=_SHARED_AUDIO_DIR,
            manifest_dir=_SHARED_MANIFEST_DIR,
        )
        if CFG.NORMALIZE_TEXT:
            norm_dir = Path(CFG.RESULTS_DIR) / "normalized_manifests"
            norm_dir.mkdir(parents=True, exist_ok=True)
            out["train"] = maybe_write_normalized_manifest(out["train"], str(norm_dir / Path(out["train"]).name))
            out["val"] = maybe_write_normalized_manifest(out["val"], str(norm_dir / Path(out["val"]).name))
            if "test" in out:
                out["test"] = maybe_write_normalized_manifest(out["test"], str(norm_dir / Path(out["test"]).name))
        return out
    ds, text_field = load_dataset_bundle(CFG.DATASET, **_dataset_loader_kwargs())
    # Audio is written to data/audio/<dataset_name>/ so clips from different
    # datasets never share a directory and WAV filenames cannot collide.
    audio_dir = os.path.join(_SHARED_AUDIO_DIR, CFG.DATASET)
    out: Dict[str, str] = {}
    out["train"] = build_nemo_manifest(ds["train"], CFG.DATASET, "train", audio_dir, _SHARED_MANIFEST_DIR, text_field)
    out["val"] = build_nemo_manifest(ds["validation"], CFG.DATASET, "val", audio_dir, _SHARED_MANIFEST_DIR, text_field)
    if "test" in ds:
        out["test"] = build_nemo_manifest(ds["test"], CFG.DATASET, "test", audio_dir, _SHARED_MANIFEST_DIR, text_field)
    if CFG.NORMALIZE_TEXT:
        norm_dir = Path(CFG.RESULTS_DIR) / "normalized_manifests"
        norm_dir.mkdir(parents=True, exist_ok=True)
        out["train"] = maybe_write_normalized_manifest(out["train"], str(norm_dir / Path(out["train"]).name))
        out["val"] = maybe_write_normalized_manifest(out["val"], str(norm_dir / Path(out["val"]).name))
        if "test" in out:
            out["test"] = maybe_write_normalized_manifest(out["test"], str(norm_dir / Path(out["test"]).name))
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
    run_dir = Path(CFG.RESULTS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = asdict(CFG)
    for k, v in list(cfg_dump.items()):
        if isinstance(v, frozenset):
            cfg_dump[k] = sorted(v)
        if k.startswith("_"):
            cfg_dump.pop(k, None)
    results: Dict[str, Any] = {"run_id": run_id, "config": cfg_dump, "run_dir": str(run_dir)}

    if CFG.RUN_ZERO_SHOT:
        results["zero_shot_val"] = evaluate_zero_shot_bundle(val_m)

    sft_ckpt = os.path.join(CFG.CHECKPOINT_DIR, "sft_model.nemo")
    sft_model, sft_metrics = run_sft_stage(
        train_m,
        val_m,
        sft_ckpt,
        run_dir / f"{run_id}_sft_epoch_metrics.csv",
        run_dir=run_dir,
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
        run_dir / f"{run_id}_rl_epoch_metrics.csv",
        run_dir=run_dir,
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

    results["debug_sample_paths"] = {
        "sft": str(run_dir / "debug" / "debug_samples_sft.jsonl"),
        "rl": str(run_dir / "debug" / "debug_samples_rl.jsonl"),
    }
    results["lightning_checkpoint_dirs"] = {"sft": str(run_dir / "checkpoints" / "sft"), "rl": str(run_dir / "checkpoints" / "rl")}

    out_path = run_dir / f"{run_id}_results.json"
    save_results_json(results, out_path)
    upload_dir_to_gcs(_SHARED_MANIFEST_DIR, CFG.UPLOAD_GCS_URI)
    upload_dir_to_gcs(str(run_dir), CFG.UPLOAD_GCS_URI)

    logger.info("Done. Key val WER: sft=%.2f rl=%.2f", sft_metrics["wer"], rl_metrics["wer"])
    return results


def run_sft_only() -> Dict[str, Any]:
    manifests = prepare_manifests()
    run_id = f"{CFG.DATASET}_seed{CFG.SEED}_sft_{int(time.time())}"
    run_dir = Path(CFG.RESULTS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _, sft_metrics = run_sft_stage(
        manifests["train"],
        manifests["val"],
        os.path.join(CFG.CHECKPOINT_DIR, "sft_model.nemo"),
        run_dir / f"{run_id}_sft_epoch_metrics.csv",
        run_dir=run_dir,
    )
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "sft": sft_metrics,
        "debug_sample_paths": {"sft": str(run_dir / "debug" / "debug_samples_sft.jsonl")},
        "lightning_checkpoint_dirs": {"sft": str(run_dir / "checkpoints" / "sft")},
    }
    save_results_json(payload, run_dir / f"{run_id}_results.json")
    upload_dir_to_gcs(str(run_dir), CFG.UPLOAD_GCS_URI)
    return sft_metrics


def run_rl_only(sft_checkpoint: str, *, resume_ckpt: Optional[str] = None) -> Dict[str, Any]:
    manifests = prepare_manifests()
    run_id = f"{CFG.DATASET}_seed{CFG.SEED}_rl_{int(time.time())}"
    run_dir = Path(CFG.RESULTS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _, rl_metrics = run_rl_stage(
        sft_checkpoint,
        manifests["train"],
        manifests["val"],
        os.path.join(CFG.CHECKPOINT_DIR, "rl_model.nemo"),
        run_dir / f"{run_id}_rl_epoch_metrics.csv",
        run_dir=run_dir,
        resume_ckpt=resume_ckpt,
    )
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "rl": rl_metrics,
        "debug_sample_paths": {"rl": str(run_dir / "debug" / "debug_samples_rl.jsonl")},
        "lightning_checkpoint_dirs": {"rl": str(run_dir / "checkpoints" / "rl")},
    }
    save_results_json(payload, run_dir / f"{run_id}_results.json")
    upload_dir_to_gcs(str(run_dir), CFG.UPLOAD_GCS_URI)
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
    p.add_argument("--debug_reward", action="store_true", help="Verbose reward compute/skip logging")
    p.add_argument("--max_audio_seconds_for_reward", type=float, default=None, help="Skip reward compute for utterances longer than this many seconds")
    p.add_argument("--debug_sample_dump", action="store_true", help="Write periodic ref/hyp samples to results/<run_id>/debug/")
    p.add_argument("--debug_sample_every_n_steps", type=int, default=None, help="Dump samples every N optimizer steps")
    p.add_argument("--debug_sample_count", type=int, default=None, help="Number of utterances per dump")
    p.add_argument("--resume_sft_ckpt", type=str, default=None, help="Resume SFT from a Lightning .ckpt (results/<run_id>/checkpoints/sft/last.ckpt)")
    p.add_argument("--resume_rl_ckpt", type=str, default=None, help="Resume RL from a Lightning .ckpt (results/<run_id>/checkpoints/rl/last.ckpt)")
    p.add_argument("--normalize_text", action="store_true", help="Normalize manifest text (lowercase, strip punctuation) to reduce tokenizer UNK collapse")
    p.add_argument("--force_fp32", action="store_true", help="Force fp32 training/eval (disables AMP)")
    p.add_argument("--grad_clip_val", type=float, default=None, help="Gradient clipping value (Lightning Trainer)")
    p.add_argument("--rl_objective", type=str, default=None, choices=["reweight_ctc", "add_penalty"], help="Stage-2 objective for reward integration")
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
    if args.max_audio_seconds_for_reward is not None:
        CFG.MAX_AUDIO_SECONDS_FOR_REWARD = float(args.max_audio_seconds_for_reward)
    if args.upload_gcs:
        CFG.UPLOAD_GCS_URI = args.upload_gcs.rstrip("/")
    CFG.USE_LORA = args.use_lora
    if args.mock_llm:
        CFG.USE_MOCK_LLM = True
    if args.real_llm:
        CFG.USE_MOCK_LLM = False
    if args.debug_reward:
        CFG.DEBUG_REWARD = True
    if args.debug_sample_dump:
        CFG.DEBUG_SAMPLE_DUMP = True
    if args.debug_sample_every_n_steps is not None:
        CFG.DEBUG_SAMPLE_EVERY_N_STEPS = int(args.debug_sample_every_n_steps)
    if args.debug_sample_count is not None:
        CFG.DEBUG_SAMPLE_COUNT = int(args.debug_sample_count)
    if args.normalize_text:
        CFG.NORMALIZE_TEXT = True
    if args.force_fp32:
        CFG.FORCE_FP32 = True
    if args.grad_clip_val is not None:
        CFG.GRAD_CLIP_VAL = float(args.grad_clip_val)
    if args.rl_objective:
        CFG.RL_OBJECTIVE = args.rl_objective
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
        # For SFT-only we currently resume by re-running the SFT stage with ckpt_path.
        # (The stage wrapper creates the trainer and uses trainer.fit(..., ckpt_path=...)).
        manifests = prepare_manifests()
        run_id = f"{CFG.DATASET}_seed{CFG.SEED}_sft_{int(time.time())}"
        run_dir = Path(CFG.RESULTS_DIR) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _, sft_metrics = run_sft_stage(
            manifests["train"],
            manifests["val"],
            os.path.join(CFG.CHECKPOINT_DIR, "sft_model.nemo"),
            run_dir / f"{run_id}_sft_epoch_metrics.csv",
            run_dir=run_dir,
            resume_ckpt=args.resume_sft_ckpt,
        )
        payload = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sft": sft_metrics,
            "debug_sample_paths": {"sft": str(run_dir / "debug" / "debug_samples_sft.jsonl")},
            "lightning_checkpoint_dirs": {"sft": str(run_dir / "checkpoints" / "sft")},
        }
        save_results_json(payload, run_dir / f"{run_id}_results.json")
        upload_dir_to_gcs(str(run_dir), CFG.UPLOAD_GCS_URI)
    elif args.stage == "rl":
        if not args.sft_checkpoint:
            raise SystemExit("--sft_checkpoint required for --stage rl")
        run_rl_only(args.sft_checkpoint, resume_ckpt=args.resume_rl_ckpt)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
