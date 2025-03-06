#! /bin/bash
#SBATCH --job-name=CVLM_pixmo       # job name

#SBATCH --partition=gpu_p2
#SBATCH -A mfl@v100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                # number of GPUs per node
#SBATCH --cpus-per-task=3           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=02:00:00              # maximum execution time (HH:MM:SS)
##SBATCH --qos=qos_gpu-t4
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/%x_%A_%a.out # output file name
#SBATCH --error=logs/%x_%A_%a.err  # error file name





set -x
cd $WORK/large_scale_uq/

module purge
module load pytorch-gpu/py3/2.3.0
mkdir -p logs

export PATH=$WORK/.local/bin:$PATH
export TORCH_HOME=$SCRATCH
export TMPDIR=$JOBSCRATCH
export DEBUG=False
export WANDB_MODE=offline

srun python lscaleuq/run.py \
 exp_name=predicted_caption+learned_cross_attention_cc12m \
 model=confidnetvlm_attention \
 dataset=pixmo \
 optimizer.lr=1e-3 \
 lr_warmup.total_iters=1 \
 backbone=clip_vit_b32 \
 engine.n_epochs=20 \
 engine.batchwise_train=true \
 engine.print_freq=1000 \
 engine.display_progress=true \
 engine.eval_zs=true \
 engine.eval_freq=1 \
 loss.weight=1.5 \
 model.keep_frozen=true \
 model.use_predicted_caption=true \
 resume=false \
 batch_size=512 \
 cluster_env=jz \
 wandb_mode=offline \
 zs_datasets=cifar \
