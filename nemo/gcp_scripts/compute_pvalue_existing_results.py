"""
Post-hoc paired bootstrap p-value for existing NeMo ASR checkpoints.

This script does not retrain. It loads two saved .nemo checkpoints, transcribes
the same manifest with both models, then computes the paired-bootstrap p-value
for the WER difference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from nemo.collections.asr.models import EncDecCTCModelBPE

import nemo_afrispeech_training as train


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute a post-hoc paired-bootstrap p-value for two ASR checkpoints")
    p.add_argument("--model_a_checkpoint", required=True, help="Baseline .nemo checkpoint, usually the SFT checkpoint")
    p.add_argument("--model_b_checkpoint", required=True, help="Comparison .nemo checkpoint, usually the RL checkpoint")
    p.add_argument("--model_a_name", default="sft", help="Label for model A in the output JSON")
    p.add_argument("--model_b_name", default="rl", help="Label for model B in the output JSON")
    p.add_argument(
        "--manifest",
        default=None,
        help="Evaluation manifest. If omitted, the script prepares/uses the validation manifest for --dataset.",
    )
    p.add_argument(
        "--dataset",
        choices=["afrispeech_clinical", "librispeech", "voxpopuli"],
        default="afrispeech_clinical",
        help="Dataset used only when --manifest is omitted.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--bootstrap_iters", type=int, default=1000)
    p.add_argument("--val_samples", type=int, default=None, help="Optional AfriSpeech validation cap")
    p.add_argument(
        "--voxpopuli_train_subset",
        type=int,
        default=None,
        help="Only used if --manifest is omitted and VoxPopuli manifests must be prepared.",
    )
    p.add_argument("--normalize_text", action="store_true", help="Use normalized manifests when preparing manifests")
    p.add_argument("--output_json", default=None, help="Where to write summary metrics and p-value")
    p.add_argument(
        "--predictions_jsonl",
        default=None,
        help="Optional JSONL output with ref/model_a/model_b predictions for later inspection.",
    )
    return p.parse_args()


def configure_training_module(args: argparse.Namespace) -> None:
    train.CFG.DATASET = args.dataset
    train.CFG.SEED = args.seed
    train.CFG.BATCH_SIZE = args.batch_size
    train.CFG.BOOTSTRAP_ITERS = args.bootstrap_iters
    train.CFG.NORMALIZE_TEXT = bool(args.normalize_text)
    if args.val_samples is not None:
        train.CFG.VAL_SAMPLES = args.val_samples
    if args.voxpopuli_train_subset is not None:
        train.CFG.VOXPOPULI_TRAIN_SUBSET = args.voxpopuli_train_subset
    train.set_seed(args.seed)


def resolve_manifest(args: argparse.Namespace) -> str:
    if args.manifest:
        return args.manifest
    default_manifest = Path(train._SHARED_MANIFEST_DIR) / f"{args.dataset}_val.json"
    if default_manifest.exists():
        return str(default_manifest)
    raise FileNotFoundError(
        f"No --manifest was provided and default validation manifest does not exist: {default_manifest}. "
        "Pass --manifest explicitly, or run the normal training/data-prep flow once to create it."
    )


def load_checkpoint(path: str) -> EncDecCTCModelBPE:
    logger.info("Loading checkpoint: %s", path)
    model = EncDecCTCModelBPE.restore_from(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def without_predictions(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metrics.items() if k not in {"_refs", "_hyps"}}


def write_predictions(path: str, refs: list[str], hyps_a: list[str], hyps_b: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ref, hyp_a, hyp_b in zip(refs, hyps_a, hyps_b):
            f.write(json.dumps({"ref": ref, "model_a_hyp": hyp_a, "model_b_hyp": hyp_b}) + "\n")
    logger.info("Wrote paired predictions: %s", out)


def main() -> None:
    args = parse_args()
    configure_training_module(args)
    manifest = resolve_manifest(args)

    model_a = load_checkpoint(args.model_a_checkpoint)
    metrics_a = train.evaluate_manifest_bundle(model_a, manifest)
    refs = metrics_a["_refs"]
    hyps_a = metrics_a["_hyps"]
    del model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_b = load_checkpoint(args.model_b_checkpoint)
    metrics_b = train.evaluate_manifest_bundle(model_b, manifest)
    hyps_b = metrics_b["_hyps"]
    del model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(refs) != len(hyps_a) or len(refs) != len(hyps_b):
        raise ValueError(
            f"Paired evaluation needs equal lengths: refs={len(refs)} "
            f"{args.model_a_name}={len(hyps_a)} {args.model_b_name}={len(hyps_b)}"
        )

    p_value = train.paired_bootstrap_wer_pvalue(refs, hyps_a, hyps_b, args.bootstrap_iters, args.seed)
    wer_delta = float(metrics_a["wer"] - metrics_b["wer"])
    payload: Dict[str, Any] = {
        "comparison": f"{args.model_a_name}_vs_{args.model_b_name}",
        "manifest": manifest,
        "model_a_name": args.model_a_name,
        "model_a_checkpoint": args.model_a_checkpoint,
        "model_a_metrics": without_predictions(metrics_a),
        "model_b_name": args.model_b_name,
        "model_b_checkpoint": args.model_b_checkpoint,
        "model_b_metrics": without_predictions(metrics_b),
        "wer_delta_model_a_minus_model_b": wer_delta,
        "paired_bootstrap_pvalue_wer": p_value,
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "n_utterances": len(refs),
        "interpretation": (
            f"Positive WER delta means {args.model_b_name} has lower WER than {args.model_a_name}; "
            "p-value is two-sided paired bootstrap over utterances."
        ),
    }

    if args.predictions_jsonl:
        write_predictions(args.predictions_jsonl, refs, hyps_a, hyps_b)
        payload["predictions_jsonl"] = args.predictions_jsonl

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote p-value summary: %s", out)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
