#!/bin/bash
#SBATCH --job-name=ConfidVLM_attn_IN_H     # job name
#SBATCH -A yxq@h100
#SBATCH --nodes=1
#SBATCH --ntasks=4                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH -C h100
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=logs/%x_%A_%a.out # output file name
#SBATCH --error=logs/%x_%A_%a.err  # error file name
#SBATCH --array=0

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

srun python lscaleuq/run.py \
 exp_name=hierarchical_imagenet_predictedT_identity_proj_attention \
 model=confidnetvlm_attention \
 dataset=imagenet_hierarchy \
 dataset.hierarchical_level=1 \
 optimizer.lr=1e-3 \
 lr_warmup.total_iters=5 \
 backbone=clip_vit_b32 \
 engine.n_epochs=400 \
 engine.template_type=imagenet_base \
 engine.eval_zs=true \
 zs_datasets=cifar \
 loss.weight=1.5 \
 model.keep_frozen=true \
 resume=true \
 batch_size=512 \
 batch_size_val=512 \
 cluster_env=jz \
 wandb_mode=offline \
