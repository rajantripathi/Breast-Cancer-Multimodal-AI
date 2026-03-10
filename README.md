# Breast Cancer Multimodal AI

Isambard-first multimodal research scaffold for breast cancer modeling. The local machine is used for editing, validation, and Git operations only. All heavy downloads, feature extraction, preprocessing, training, and evaluation are designed to run on Isambard.

## Components

- `agents/`: modality wrappers, including the Phase 1 vision foundation-model registry and feature extractor
- `training/`: modality trainers plus late-fusion verifier trainer
- `data/`: remote-first download, preprocessing, and split generation scripts
- `slurm/`: end-to-end Isambard job scripts
- `orchestrator/`: unified inference entrypoint
- `apps/`: lightweight Streamlit demo that reads exported artifacts
- `tasks/`: TODO tracking, experiment log, and lessons learned

## Local bootstrap

```bash
cd ~/Projects/Breast-Cancer-Multimodal-AI
bash setup.sh
python -m py_compile $(find . -name '*.py')
```

Local validation is intentionally lightweight. It uses seeded sample cases and deterministic fallback feature extraction rather than downloading large assets or running foundation models.

## Isambard bootstrap

```bash
git clone https://github.com/rajantripathi/Breast-Cancer-Multimodal-AI.git
cd Breast-Cancer-Multimodal-AI
cp .env.example .env
sbatch slurm/00_setup.sh
```

## Pipeline

```bash
export SMOKE_TEST=1   # optional for short verification jobs
bash run_full_pipeline.sh
```

The pipeline submits setup, model download, data download, preprocessing, vision feature extraction, parallel modality training, verifier training, and evaluation jobs. The Streamlit app is intended for local or tunneled use against exported artifacts.

## Direct module entrypoints

```bash
python -m data.download.download_vision
python -m data.preprocess.preprocess_vision
python -m agents.vision.extract_features --model uni2 --smoke-test
python -m training.vision_trainer --model-key uni2 --smoke-test
python -m training.verifier_trainer
python -m orchestrator.run --all-sample-cases
python -m evaluation.evaluate
```

## Security

Do not commit real secrets. Rotate any previously pasted GitHub, Hugging Face, or Kaggle tokens before real Isambard use.

## Phase 1 Vision Upgrade

The repo now includes a Phase 1 vision-agent migration:

- `agents/vision/foundation_models.py` is the single source of truth for supported vision backbones
- `agents/vision/extract_features.py` generates per-sample feature artifacts from a manifest
- `training/vision_trainer.py` trains from extracted embeddings rather than raw image-text placeholders
- `config/default.yaml` and `config/isambard.yaml` carry the vision registry and extraction defaults
