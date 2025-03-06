PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES="0" python lscaleuq/run.py \
 exp_name="uqvlm_backbone_cifar10" \
 dataset=cifar10 \
 backbone=clip_vit_l14 \
 zs_datasets=data_suite \
 model=confidnetvlm_attention \
 model.keep_frozen=true \
 model.n_iter_freeze_proj=1000 \
 model.use_predicted_caption=true \
 model.use_attention=true \
 engine.n_epochs=500 \
 engine.batchwise_train=false \
 engine.eval_zs=false \
 engine.eval_only=false \
 resume=False \
 loss.weight=10 \
 optimizer.lr=0.01 \
 batch_size=128 \
 cluster_env=cnam \
 wandb_mode=online \
 debug=false \



PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES="0" python lscaleuq/run.py \
 exp_name="uqvlm_backbone_cifar10" \
 dataset=cifar10 \
 backbone=clip_vit_b16 \
 zs_datasets=data_suite \
 model=confidnetvlm_attention \
 model.keep_frozen=true \
 model.n_iter_freeze_proj=1000 \
 model.use_predicted_caption=true \
 model.use_attention=true \
 engine.n_epochs=500 \
 engine.batchwise_train=false \
 engine.eval_zs=false \
 engine.eval_only=false \
 resume=False \
 loss.weight=10 \
 optimizer.lr=0.01 \
 batch_size=128 \
 cluster_env=cnam \
 wandb_mode=online \
 debug=false \