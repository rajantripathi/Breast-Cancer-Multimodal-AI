#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SBATCH_EXPORT="ALL,REPO_DIR=$ROOT_DIR,SMOKE_TEST=${SMOKE_TEST:-0}"

setup_job=$(sbatch --parsable --export="$SBATCH_EXPORT" slurm/00_setup.sh)
model_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$setup_job" slurm/01_download_base_models.sh)
data_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$setup_job" slurm/02_download_data.sh)
prep_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$data_job":"$model_job" slurm/03_preprocess.sh)

vision_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$prep_job" slurm/04_train_vision.sh)
ehr_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$prep_job" slurm/05_train_ehr.sh)
genomics_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$prep_job" slurm/06_train_genomics.sh)
literature_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$prep_job" slurm/07_train_literature.sh)

verifier_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$vision_job":"$ehr_job":"$genomics_job":"$literature_job" slurm/08_train_verifier.sh)
eval_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$verifier_job" slurm/09_evaluate.sh)
ui_job=$(sbatch --parsable --export="$SBATCH_EXPORT" --dependency=afterok:"$eval_job" slurm/10_deploy_ui.sh)

echo "Submitted jobs:"
echo "  setup:      $setup_job"
echo "  models:     $model_job"
echo "  data:       $data_job"
echo "  preprocess: $prep_job"
echo "  vision:     $vision_job"
echo "  ehr:        $ehr_job"
echo "  genomics:   $genomics_job"
echo "  literature: $literature_job"
echo "  verifier:   $verifier_job"
echo "  evaluate:   $eval_job"
echo "  ui:         $ui_job"
