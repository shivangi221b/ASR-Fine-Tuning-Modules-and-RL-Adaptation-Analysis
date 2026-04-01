# ESPNet — ASR Domain Adaptation (forthcoming)

This folder will contain all ESPNet experiments for the shared ASR domain-adaptation study, using the same datasets and evaluation protocol as the NeMo experiments in [`../nemo/`](../nemo/).

See the [root README](../README.md) for overall goals and cross-framework progress.

## Planned work

- [ ] ESPNet environment setup and `requirements-espnet-train.txt`
- [ ] Data preparation scripts that consume manifests from [`../data/`](../data/)
- [ ] SFT baseline training recipe (conformer or transformer encoder)
- [ ] Reward-augmented fine-tuning (MWER / WWER / LLM) matching the NeMo reward modes
- [ ] Evaluation on AfriSpeech-200 clinical, VoxPopuli, and LibriSpeech (catastrophic forgetting)
- [ ] Results JSON schema compatible with the NeMo output for side-by-side comparison

## Datasets

Same three datasets as NeMo — see [`../data/README.md`](../data/README.md) for details.
