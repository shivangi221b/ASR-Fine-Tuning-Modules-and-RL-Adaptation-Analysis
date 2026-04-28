"""
data/loaders.py
===============
Framework-agnostic HuggingFace dataset loaders for the three shared datasets.

All functions return a (DatasetDict, text_field) pair where:
  - DatasetDict always contains at least "train" and "validation" splits
  - text_field is the column name that holds the ground-truth transcript

These loaders carry no dependency on any framework config object; every
tunable parameter is an explicit argument with a documented default.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from datasets import Audio, Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def require_datasets_script_support() -> None:
    """HuggingFace ``datasets`` 3+ no longer runs hub dataset ``.py`` scripts (e.g. AfriSpeech)."""
    import datasets as hf_datasets

    parts = hf_datasets.__version__.split(".")
    major = int(parts[0]) if parts[0].isdigit() else 0
    if major >= 3:
        raise RuntimeError(
            f"AfriSpeech-200 needs `datasets` < 3 (dataset scripts). You have {hf_datasets.__version__}. "
            'Run: python -m pip install "datasets>=2.14.0,<3.0.0"'
        )


# ---------------------------------------------------------------------------
# AfriSpeech-200 (clinical)
# ---------------------------------------------------------------------------

def _resolve_afrispeech_name() -> str:
    """Return whichever HF repo name works in the current environment."""
    try:
        load_dataset("intronhealth/afrispeech-200", "all", split="validation", streaming=True)
        return "intronhealth/afrispeech-200"
    except Exception:
        return "tobiolatunji/afrispeech-200"


def _collect_clinical_from_stream(stream, max_n: Optional[int]) -> List[dict]:
    out: List[dict] = []
    for sample in stream:
        if str(sample.get("domain", "")).lower() != "clinical":
            continue
        if not str(sample.get("transcript", "")).strip():
            continue
        out.append(sample)
        if max_n is not None and len(out) >= max_n:
            break
    return out


def load_afrispeech_clinical(
    train_n: Optional[int] = None,
    val_n: Optional[int] = None,
    test_n: Optional[int] = None,
    load_test: bool = True,
) -> Tuple[DatasetDict, str]:
    """
    Load AfriSpeech-200 clinical splits via HuggingFace streaming.

    Parameters
    ----------
    train_n : int | None
        Cap on training samples after clinical filter (None = all).
    val_n : int | None
        Cap on validation samples (None = all).
    test_n : int | None
        Cap on test samples (None = all). Ignored when load_test=False.
    load_test : bool
        Whether to load the test split at all.

    Returns
    -------
    (DatasetDict, "transcript")
    """
    require_datasets_script_support()
    name = _resolve_afrispeech_name()
    logger.info("Loading AfriSpeech-200 clinical from %s", name)

    logger.info("  Streaming TRAIN (clinical) …")
    train_samples = _collect_clinical_from_stream(
        load_dataset(name, "all", split="train", streaming=True), train_n
    )
    logger.info("  Train clinical clips: %d (cap=%s)", len(train_samples), train_n)

    logger.info("  Streaming VALIDATION (clinical) …")
    val_samples = _collect_clinical_from_stream(
        load_dataset(name, "all", split="validation", streaming=True), val_n
    )
    logger.info("  Val clinical clips: %d (cap=%s)", len(val_samples), val_n)

    test_samples: List[dict] = []
    if load_test:
        logger.info("  Streaming TEST (clinical) …")
        test_samples = _collect_clinical_from_stream(
            load_dataset(name, "all", split="test", streaming=True), test_n
        )
        logger.info("  Test clinical clips: %d (cap=%s)", len(test_samples), test_n)

    train_ds = Dataset.from_list(train_samples).cast_column("audio", Audio(sampling_rate=16_000))
    val_ds = Dataset.from_list(val_samples).cast_column("audio", Audio(sampling_rate=16_000))
    splits = {"train": train_ds, "validation": val_ds}
    if test_samples:
        splits["test"] = Dataset.from_list(test_samples).cast_column("audio", Audio(sampling_rate=16_000))

    return DatasetDict(splits), "transcript"


# ---------------------------------------------------------------------------
# LibriSpeech clean-100
# ---------------------------------------------------------------------------

def load_librispeech(
    train_n: int = 5_000,
    val_n: int = 2_703,
) -> Tuple[DatasetDict, str]:
    """
    Load a slice of LibriSpeech clean-100.

    Parameters
    ----------
    train_n : int
        Number of training samples to take (sequential slice).
    val_n : int
        Number of validation samples to take.

    Returns
    -------
    (DatasetDict, "text")
    """
    logger.info("Loading LibriSpeech clean-100 (train=%d, val=%d) …", train_n, val_n)
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split={
            "train": f"train.100[:{train_n}]",
            "validation": f"validation[:{val_n}]",
        },
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    logger.info("  LibriSpeech train=%d val=%d", len(ds["train"]), len(ds["validation"]))
    return ds, "text"


def load_librispeech_eval(val_n: int = 2_703) -> Tuple[Dataset, str]:
    """
    Load only the LibriSpeech validation split (used for catastrophic-forgetting eval).

    Returns
    -------
    (Dataset, "text")   — a single split, not a DatasetDict
    """
    logger.info("Loading LibriSpeech eval split (val_n=%d) …", val_n)
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split={"validation": f"validation[:{val_n}]"},
    )
    val_ds = ds["validation"].cast_column("audio", Audio(sampling_rate=16_000))
    logger.info("  LibriSpeech eval clips: %d", len(val_ds))
    return val_ds, "text"


# ---------------------------------------------------------------------------
# VoxPopuli (English)
# ---------------------------------------------------------------------------

def load_voxpopuli(
    train_n: Optional[int] = None,
    val_n: int = 1_750,
    seed: int = 42,
) -> Tuple[DatasetDict, str]:
    """
    Load VoxPopuli English.

    Parameters
    ----------
    train_n : int | None
        Number of training utterances, sampled without replacement using seed.
        None = full training split.
    val_n : int
        Number of validation utterances (sequential slice).
    seed : int
        Random seed for reproducible training-set sampling.

    Returns
    -------
    (DatasetDict, "normalized_text")
    """
    logger.info("Loading VoxPopuli English (train_n=%s, val_n=%d, seed=%d) …", train_n, val_n, seed)
    ds_train_full = load_dataset("facebook/voxpopuli", "en", split="train")
    if train_n is None:
        train_ds = ds_train_full
    else:
        rng = np.random.RandomState(seed)
        n_take = min(train_n, len(ds_train_full))
        indices = rng.choice(len(ds_train_full), size=n_take, replace=False)
        train_ds = ds_train_full.select(indices.tolist())

    val_ds = load_dataset("facebook/voxpopuli", "en", split=f"validation[:{val_n}]")

    ds = DatasetDict({"train": train_ds, "validation": val_ds})
    ds = DatasetDict({k: v.cast_column("audio", Audio(sampling_rate=16_000)) for k, v in ds.items()})
    logger.info("  VoxPopuli train=%d val=%d", len(ds["train"]), len(ds["validation"]))
    return ds, "normalized_text"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_dataset_bundle(
    name: str,
    **kwargs,
) -> Tuple[DatasetDict, str]:
    """
    Convenience dispatcher.

    name : "afrispeech_clinical" | "librispeech" | "voxpopuli"
    **kwargs forwarded to the corresponding loader function.
    """
    if name == "afrispeech_clinical":
        return load_afrispeech_clinical(**kwargs)
    if name == "librispeech":
        return load_librispeech(**kwargs)
    if name == "voxpopuli":
        return load_voxpopuli(**kwargs)
    raise ValueError(
        f"Unknown dataset '{name}'. "
        "Choose from: afrispeech_clinical, librispeech, voxpopuli."
    )
