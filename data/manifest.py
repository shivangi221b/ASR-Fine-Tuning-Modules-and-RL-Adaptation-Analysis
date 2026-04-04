"""
data/manifest.py
================
Writes audio clips to disk and produces a NeMo-format JSON manifest
(one JSON object per line: audio_filepath, duration, text).

The same manifest files can be reused by ESPNet's data-prep scripts
once those are added, since WAV + a simple JSONL index is toolkit-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import soundfile as sf

logger = logging.getLogger(__name__)

# Utterance duration guard-rails (seconds).  Clips outside this range are
# skipped to avoid very short/long outliers causing training instability.
MIN_DURATION_S: float = 0.5
MAX_DURATION_S: float = 30.0


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
