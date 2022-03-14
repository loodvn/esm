#! /bin/bash

#!/bin/bash
#SBATCH -c 2                               # Request two cores
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-23:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad,gpu_marks,gpu          #,gpu_requeue        # Partition to run in
# If on gpu_quad, use teslaV100s
# If on gpu_requeue, use teslaM40 or a100?
# If on gpu, any of them are fine (teslaV100, teslaM40, teslaK80) although K80 sometimes is too slow
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_doublep
#SBATCH --qos=gpuquad_qos
#SBATCH --mem=20G                          # Memory total in MB (for all cores)

#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user="lodevicus_vanniekerk@hms.harvard.edu"

#SBATCH --output=logs/slurm_files/slurm-%j-%x.out                 # File to which STDOUT + STDERR will be written, including job ID in filename
##SBATCH --error=logs/slurm_files/slurm-%j-%x.err                 # Optional: include error in separate file
#SBATCH --job-name="tmp_hydra_template"

# Job array-specific
##SBATCH --output=slurm_files/slurm-lvn-%A_%3a-%x.out   # Nice tip: using %3a to pad to 3 characters (23 -> 023)
##SBATCH --error=slurm_files/slurm-lvn-%A_%3a-%x.err   # Optional: Redirect STDERR to its own file
##SBATCH --array=0  		    # Job arrays, range inclusive (MIN-MAX%MAX_CONCURRENT_TASKS)

# Quite neat workflow:
# Submit job array in held state, then release first job to test
# Add a dependency so that the next jobs are submitted as soon as the first job completes successfully:
# scontrol update Dependency=afterok:<jobid>_0 JobId=<jobid>
# Release all the other jobs; they'll be stuck until the first job is done
# Other tips:
#  Don't include any commands in-between the #SBATCH directives because then the following options will be ignored
################################################################################

set -e # fail fully on first line failure (from Joost slurm_for_ml)

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
echo "GPU available: $(nvidia-smi)"


#SBATCH --job-name="esm_embeddings"


# Make prints more stable
export PYTHONUNBUFFERED=1

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

# This is secret and shouldn't be checked into version control
# Stored in .netrc
# Name and notes optional
export WANDB_NAME="esm_extract"
#mkdir -p $HOME/wandb  # Just log in current directory by default
#export WANDB_DIR=$HOME
#export TMPDIR=$HOME/tmp
#mkdir -p $TMPDIR
export WANDB_NOTES="Extracting embeddings from ESM"

# O2
module load gcc/6.2.0 miniconda3/4.10.3 cuda/11.2

#conda env update -f environment.yml
conda activate torch  # Note: Will have to manually update this env with new requirements

# Monitor GPU usage (store outputs in ./gpu_logs/)
~/job_gpu_monitor.sh --interval 1m gpu_logs &

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