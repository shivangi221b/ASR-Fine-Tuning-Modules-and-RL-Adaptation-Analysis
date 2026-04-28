"""
Post-hoc checkpoint evaluator.

Evaluates one or more saved .nemo checkpoints against any manifest and
writes per-checkpoint metrics to JSON. Designed for two recurring needs:

  1. LibriSpeech-after-RL  — catastrophic forgetting check
     python3 eval_checkpoints.py \\
       --manifest data/manifests/librispeech_forgetting_eval.json \\
       --eval_name librispeech_forgetting \\
       --checkpoints afrispeech_mwer_rl:checkpoints/afrispeech_clinical_...rl_mwer.nemo \\
                     afrispeech_wwer_rl:checkpoints/rl_model.nemo \\
                     voxpopuli_mwer_rl:checkpoints/voxpopuli_..._rl_mwer.nemo \\
                     voxpopuli_wwer_rl:checkpoints/voxpopuli_..._rl_wwer.nemo \\
       --output_dir results/forgetting_eval/

  2. AfriSpeech test-set RL evaluation (test_rl)
     python3 eval_checkpoints.py \\
       --manifest data/manifests/afrispeech_clinical_test.json \\
       --eval_name afrispeech_test_rl \\
       --checkpoints afrispeech_sft:checkpoints/sft_model.nemo \\
                     afrispeech_mwer_rl:checkpoints/afrispeech_clinical_...rl_mwer.nemo \\
                     afrispeech_wwer_rl:checkpoints/rl_model.nemo \\
       --output_dir results/test_eval/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from nemo.collections.asr.models import EncDecCTCModelBPE

import nemo_afrispeech_training as train


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multiple .nemo checkpoints against a single manifest (no retraining)."
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help=(
            "One or more checkpoints in NAME:PATH format. "
            "NAME appears as the key in the output JSON. "
            "Example: sft:checkpoints/sft_model.nemo mwer_rl:checkpoints/rl_mwer.nemo"
        ),
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="Path to the JSONL manifest to evaluate against.",
    )
    p.add_argument(
        "--eval_name",
        default="eval",
        help="Short label for this evaluation run (used in output filenames).",
    )
    p.add_argument(
        "--output_dir",
        default="results/posthoc_eval",
        help="Directory to write result JSONs into.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dataset",
        choices=["afrispeech_clinical", "librispeech", "voxpopuli"],
        default="afrispeech_clinical",
        help="Used only to select the correct domain-term vocabulary for EWER/F1 metrics.",
    )
    return p.parse_args()


def parse_checkpoint_specs(specs: List[str]) -> List[Tuple[str, str]]:
    """Parse 'NAME:PATH' strings into (name, path) tuples."""
    result = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"Checkpoint spec must be in NAME:PATH format, got: {spec!r}. "
                "Example: sft:checkpoints/sft_model.nemo"
            )
        name, _, path = spec.partition(":")
        result.append((name.strip(), path.strip()))
    return result


def load_checkpoint(path: str) -> EncDecCTCModelBPE:
    logger.info("Loading checkpoint: %s", path)
    model = EncDecCTCModelBPE.restore_from(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def without_internal(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def main() -> None:
    args = parse_args()

    train.CFG.DATASET = args.dataset
    train.CFG.BATCH_SIZE = args.batch_size
    train.set_seed(args.seed)

    checkpoint_specs = parse_checkpoint_specs(args.checkpoints)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest
    if not Path(manifest).exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}\n"
            "For LibriSpeech forgetting eval, the manifest is created during the first full pipeline run "
            "(librispeech_forgetting_eval.json). If it does not exist yet, run:\n\n"
            "  python3 nemo_afrispeech_training.py --stage sft --dataset librispeech --smoke_test\n\n"
            "or copy the manifest from a previous run. For AfriSpeech test, use "
            "data/manifests/afrispeech_clinical_test.json."
        )

    all_results: Dict[str, Any] = {
        "eval_name": args.eval_name,
        "manifest": manifest,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "results": {},
    }

    for name, ckpt_path in checkpoint_specs:
        logger.info("=" * 60)
        logger.info("Evaluating checkpoint: %s  (%s)", name, ckpt_path)
        logger.info("=" * 60)

        model = load_checkpoint(ckpt_path)
        metrics = train.evaluate_manifest_bundle(model, manifest)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        clean = without_internal(metrics)
        all_results["results"][name] = {
            "checkpoint": ckpt_path,
            **clean,
        }

        per_ckpt_path = out_dir / f"{args.eval_name}_{name}.json"
        with per_ckpt_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "eval_name": args.eval_name,
                    "checkpoint_name": name,
                    "checkpoint_path": ckpt_path,
                    "manifest": manifest,
                    "dataset": args.dataset,
                    **clean,
                },
                f,
                indent=2,
            )
        logger.info("  -> wrote %s", per_ckpt_path)

    summary_path = out_dir / f"{args.eval_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary written to: %s", summary_path)

    logger.info("\n=== SUMMARY: %s ===", args.eval_name)
    for name, r in all_results["results"].items():
        logger.info("  %-25s  WER=%.3f%%  CER=%.3f%%  EWER=%.3f%%", name, r["wer"], r["cer"], r["ewer"])


if __name__ == "__main__":
    main()
