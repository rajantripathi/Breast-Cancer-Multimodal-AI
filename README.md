# Breast Cancer Multimodal AI

Isambard-first research scaffold for breast cancer multimodal modeling. The local machine is used for editing and Git operations only. All heavy downloads, preprocessing, training, and evaluation are designed to run on Isambard.

## Components

- `agents/`: modality wrappers for vision, clinical tabular, genomics, and literature inputs
- `training/`: modality trainers plus late-fusion verifier trainer
- `data/`: remote-first download, preprocessing, and split generation scripts
- `slurm/`: end-to-end Isambard job scripts
- `orchestrator/`: unified inference entrypoint
- `apps/`: lightweight Streamlit demo that reads exported artifacts

## Local bootstrap

```bash
cd ~/Projects/Breast-Cancer-Multimodal-AI
bash setup.sh
python -m py_compile $(find . -name '*.py')
```

Local validation is intentionally lightweight. It uses the seeded sample cases and tiny fallback datasets rather than downloading large assets.

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

The pipeline submits setup, model download, data download, preprocessing, parallel modality training, verifier training, and evaluation jobs. The Streamlit app is intended for local or tunneled use against exported artifacts.

## Direct module entrypoints

```bash
python -m data.download.download_vision
python -m data.preprocess.preprocess_vision
python -m training.vision_trainer
python -m training.verifier_trainer
python -m orchestrator.run --all-sample-cases
python -m evaluation.evaluate
```

## Security

Do not commit real secrets. Rotate any previously pasted GitHub, Hugging Face, or Kaggle tokens before real Isambard use.
