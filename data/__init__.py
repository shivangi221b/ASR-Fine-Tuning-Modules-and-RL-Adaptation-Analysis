"""
data
====
Shared dataset loading and manifest utilities for all ASR framework experiments.

Public API
----------
Loaders (return (DatasetDict, text_field)):
    load_afrispeech_clinical  — AfriSpeech-200 filtered to domain=="clinical"
    load_librispeech          — LibriSpeech clean-100 train + validation slice
    load_librispeech_eval     — LibriSpeech validation only (forgetting eval)
    load_voxpopuli            — VoxPopuli English random train subset
    load_dataset_bundle       — Dispatcher by dataset name

Manifest builders:
    build_nemo_manifest                        — Decode audio → WAV files + NeMo JSONL manifest
    prepare_afrispeech_clinical_manifests_streaming — Low-RAM streaming path for AfriSpeech clinical
    prepare_voxpopuli_manifests_streaming      — Low-disk streaming path for VoxPopuli
"""

from .loaders import (
    load_afrispeech_clinical,
    load_dataset_bundle,
    load_librispeech,
    load_librispeech_eval,
    load_voxpopuli,
)
from .manifest import (
    build_nemo_manifest,
    prepare_afrispeech_clinical_manifests_streaming,
    prepare_voxpopuli_manifests_streaming,
)

__all__ = [
    "load_afrispeech_clinical",
    "load_dataset_bundle",
    "load_librispeech",
    "load_librispeech_eval",
    "load_voxpopuli",
    "build_nemo_manifest",
    "prepare_afrispeech_clinical_manifests_streaming",
    "prepare_voxpopuli_manifests_streaming",
]
