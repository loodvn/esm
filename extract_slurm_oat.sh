#! /bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=./slurm_stdout/slurm-lvn-%j.out
#SBATCH --error=./slurm_stdout/slurm-lvn-%j.err
#SBATCH --job-name="esm_embeddings"

##SBATCH --nodelist=oat0
##SBATCH --partition=preemptible

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user="lood.vanniekerk@cs.ox.ac.uk"

# Extra slurm goodies: https://slurm.schedmd.com/sbatch.html
# Time to run at e.g. night. e.g. "now+1hour" The value may be changed after job submission using the scontrol command.
##SBATCH --begin=HH:MM:SS
# Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
##SBATCH --time=1-0
##SBATCH --deadline=now+6hour
##SBATCH --nodes=min-max
# Job array-specific
##SBATCH --output=./slurm_stdout/slurm-lvn-%A_%a.out


# Make prints more stable
export PYTHONUNBUFFERED=1

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

# WANDB_API_KEY in .netrc
# Name and notes optional
export WANDB_NAME="esm_filtered_len_again"
export WANDB_DIR=$HOME
mkdir -p $HOME/wandb
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
export WANDB_NOTES="First try WandB from slurm"

# used to be /scratch-ssd/oatml/scripts/run_locked.sh
/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f virus_env.yml
source /scratch-ssd/oatml/miniconda3/bin/activate virus

srun echo "test_stdout" && \
    pwd && \
    python esm/extract.py \
    esm1b_t33_650M_UR50S \
    data/virscan/vir2.sequence_labels_filtered_len.fasta \
    virus_emb_esm1b_filtered_len/ \
    --repr_layers 33 \
    --include per_tok \
    --truncate
#wandb init && \
