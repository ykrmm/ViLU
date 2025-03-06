#!/bin/bash
#SBATCH --job-name=CVLM_CC12M        # job name
#SBATCH -A yxq@h100
#SBATCH --nodes=2
#SBATCH --ntasks=2                   # number of MP tasks
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
 exp_name=predicted_caption+learned_cross_attention_cc12m \
 model=confidnetvlm_attention \
 dataset=cc12m \
 optimizer.lr=1e-3 \
 lr_warmup.total_iters=1 \
 backbone=clip_vit_b32 \
 engine.n_epochs=300 \
 engine.batchwise_train=true \
 engine.print_freq=1000 \
 engine.display_progress=true \
 engine.eval_zs=false \
 loss.weight=1.5 \
 model.keep_frozen=true \
 model.n_iter_freeze_proj=10000 \
 model.use_predicted_caption=true \
 resume=true \
 batch_size=512 \
 cluster_env=jz \
 wandb_mode=offline \
