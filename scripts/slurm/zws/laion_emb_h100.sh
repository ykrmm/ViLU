#!/bin/bash
#SBATCH --job-name=CVLM_LAION_MULTI        # job name
#SBATCH -A zws@h100
#SBATCH --nodes=4
#SBATCH --ntasks=4                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH -C h100
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=99:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --output=logs/%x_%A_%a.out # output file name
#SBATCH --error=logs/%x_%A_%a.err  # error file name
#SBATCH --array=0-4

set -x
cd $WORK/large_scale_uq/

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1
mkdir -p logs

export PATH=$WORK/.local/bin:$PATH
export TORCH_HOME=$SCRATCH
export TMPDIR=$JOBSCRATCH
export DEBUG=False
export WANDB_MODE=offline


lr=(0.0001 0.001 0.005 0.01 0.05)

srun python lscaleuq/run.py \
 exp_name="confidnet_laion" \
 dataset=laion_emb \
 backbone=clip_vit_b32 \
 zs_datasets=data_suite \
 model=confidnetvlm_attention \
 model.keep_frozen=true \
 model.use_predicted_caption=true \
 engine.n_epochs=40 \
 engine.batchwise_train=true \
 engine.eval_zs=true \
 engine.eval_only=false \
 engine.eval_freq=1 \
 engine.eval_zs_freq=1 \
 engine.use_gather=false \
 resume=False \
 loss.weight=1.5 \
 optimizer.lr=${lr[$SLURM_ARRAY_TASK_ID]} \
 batch_size=1024 \
 cluster_env=jz \
 wandb_mode=offline \
 debug=false