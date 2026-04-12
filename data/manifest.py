"""
data/manifest.py
================
Writes audio clips to disk and produces a NeMo-format JSON manifest
(one JSON object per line: audio_filepath, duration, text).

The same manifest files can be reused by ESPNet's data-prep scripts
once those are added, since WAV + a simple JSONL index is toolkit-agnostic.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Dict, Optional

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset

from .loaders import _resolve_afrispeech_name, require_datasets_script_support

logger = logging.getLogger(__name__)

# Utterance duration guard-rails (seconds).  Clips outside this range are
# skipped to avoid very short/long outliers causing training instability.
MIN_DURATION_S: float = 0.5
MAX_DURATION_S: float = 30.0


def _mono_float32(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    return np.ascontiguousarray(x.reshape(-1), dtype=np.float32)


def _resample_to_16k(arr: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == 16000:
        return arr, 16000
    y = librosa.resample(np.asarray(arr, dtype=np.float32), orig_sr=sr, target_sr=16000)
    return y.astype(np.float32), 16000


def _stream_afrispeech_clinical_split_to_manifest(
    hf_name: str,
    hf_split: str,
    split_name: str,
    dataset_name: str,
    audio_dir: str,
    manifest_dir: str,
    max_clinical: Optional[int],
    min_duration: float = MIN_DURATION_S,
    max_duration: float = MAX_DURATION_S,
) -> str:
    """
    One pass over a HuggingFace streaming split: filter clinical, resample to 16 kHz,
    write WAV + JSONL manifest lines. Peak RAM stays O(1) in split size.
    """
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"{dataset_name}_{split_name}.json")
    logger.info(
        "  [stream] AfriSpeech clinical HF split=%s -> manifest %s (cap=%s) …",
        hf_split,
        manifest_path,
        max_clinical,
    )
    stream = load_dataset(hf_name, "all", split=hf_split, streaming=True)
    written = 0
    skipped = 0
    hf_rows = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for sample in stream:
            hf_rows += 1
            if max_clinical is not None and written >= max_clinical:
                logger.info(
                    "  [stream] %s: reached clinical cap %s after scanning %d HF rows",
                    split_name,
                    max_clinical,
                    hf_rows,
                )
                break
            if str(sample.get("domain", "")).lower() != "clinical":
                continue
            text = str(sample.get("transcript", "")).strip().lower()
            if not text:
                skipped += 1
                continue
            audio = sample.get("audio")
            if not audio or "array" not in audio:
                skipped += 1
                continue
            try:
                arr = np.asarray(audio["array"])
                sr = int(audio["sampling_rate"])
            except (TypeError, ValueError, KeyError) as e:
                logger.debug("skip row (audio decode): %s", e)
                skipped += 1
                continue
            arr = _mono_float32(arr)
            arr, sr = _resample_to_16k(arr, sr)
            duration = float(len(arr)) / float(sr)
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue
            wav_path = os.path.join(audio_dir, f"{split_name}_{written:06d}.wav")
            sf.write(wav_path, arr, 16000)
            entry = {
                "audio_filepath": os.path.abspath(wav_path),
                "text": text,
                "duration": round(duration, 3),
            }
            mf.write(json.dumps(entry) + "\n")
            written += 1
            if written % 500 == 0:
                logger.info(
                    "    [stream] %s: clinical_written=%d hf_rows_scanned=%d skipped=%d",
                    split_name,
                    written,
                    hf_rows,
                    skipped,
                )
    logger.info(
        "  [stream] Manifest %s: %d rows (%d skipped, %d HF rows scanned)",
        manifest_path,
        written,
        skipped,
        hf_rows,
    )
    return manifest_path


def prepare_afrispeech_clinical_manifests_streaming(
    train_n: Optional[int],
    val_n: Optional[int],
    test_n: Optional[int],
    load_test: bool,
    dataset_name: str,
    audio_base_dir: str,
    manifest_dir: str,
) -> Dict[str, str]:
    """Build train/val/(optional test) manifests without materializing full HF splits in RAM."""
    require_datasets_script_support()
    hf_name = _resolve_afrispeech_name()
    audio_dir = os.path.join(audio_base_dir, dataset_name)
    out: Dict[str, str] = {}
    logger.info(
        "Using streaming AfriSpeech manifest path (stream -> WAV + manifest); "
        "peak RAM should stay far below Dataset.from_list."
    )
    out["train"] = _stream_afrispeech_clinical_split_to_manifest(
        hf_name, "train", "train", dataset_name, audio_dir, manifest_dir, train_n
    )
    gc.collect()
    out["val"] = _stream_afrispeech_clinical_split_to_manifest(
        hf_name, "validation", "val", dataset_name, audio_dir, manifest_dir, val_n
    )
    gc.collect()
    if load_test:
        out["test"] = _stream_afrispeech_clinical_split_to_manifest(
            hf_name, "test", "test", dataset_name, audio_dir, manifest_dir, test_n
        )
        gc.collect()
    return out


def build_nemo_manifest(
    dataset,
    dataset_name: str,
    split_name: str,
    audio_dir: str,
    manifest_dir: str,
    text_field: str,
    min_duration: float = MIN_DURATION_S,
    max_duration: float = MAX_DURATION_S,
) -> str:
    """
    Decode audio from a HuggingFace Dataset split, write WAV files, and
    produce a NeMo-format JSON manifest.

    The output manifest is named ``{dataset_name}_{split_name}.json`` so that
    manifests for different datasets can coexist in the same directory without
    collision (e.g. ``afrispeech_clinical_train.json`` and
    ``voxpopuli_train.json`` both live in ``data/manifests/``).

    WAV files are written as ``{split_name}_{index:06d}.wav`` inside
    *audio_dir*, which should already be scoped per-dataset by the caller
    (e.g. ``data/audio/afrispeech_clinical/``).

    Parameters
    ----------
    dataset : datasets.Dataset
        A single split (not a DatasetDict).
    dataset_name : str
        Dataset identifier used as the filename prefix, e.g.
        ``"afrispeech_clinical"``, ``"voxpopuli"``, ``"librispeech"``.
    split_name : str
        Split label used in the filename, e.g. ``"train"``, ``"val"``,
        ``"test"``, ``"forgetting_eval"``.
    audio_dir : str
        Directory where WAV files are written.  Should be dataset-scoped to
        avoid filename collisions across datasets.
    manifest_dir : str
        Directory where the manifest file is written.  Shared across datasets;
        collision is prevented by the ``dataset_name`` prefix in the filename.
    text_field : str
        Column name in *dataset* that holds the ground-truth transcript.
    min_duration : float
        Utterances shorter than this (seconds) are skipped.
    max_duration : float
        Utterances longer than this (seconds) are skipped.

    Returns
    -------
    str
        Absolute path to the written manifest file.
    """
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"{dataset_name}_{split_name}.json")

    written = 0
    skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for i, sample in enumerate(dataset):
            try:
                audio_array = sample["audio"]["array"]
                sr = int(sample["audio"]["sampling_rate"])
            except (KeyError, TypeError):
                skipped += 1
                continue

            text = str(sample.get(text_field, "")).strip().lower()
            if not text:
                skipped += 1
                continue

            wav_path = os.path.join(audio_dir, f"{split_name}_{i:06d}.wav")
            sf.write(wav_path, audio_array, sr)
            duration = len(audio_array) / float(sr)

            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue

            entry = {
                "audio_filepath": os.path.abspath(wav_path),
                "text": text,
                "duration": round(duration, 3),
            }
            fh.write(json.dumps(entry) + "\n")
            written += 1

    logger.info(
        "Manifest %s: %d rows written, %d skipped",
        manifest_path,
        written,
        skipped,
    )
    return manifest_path
