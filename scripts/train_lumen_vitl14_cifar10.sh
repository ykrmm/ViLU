PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES="0" python lumen/run.py \
 exp_name="lumen_cifar10" \
 dataset=cifar10 \
 backbone=clip_vit_l14 \
 model=lumen_attention \
 model.keep_frozen=false \
 model.n_iter_freeze_proj=1000 \
 model.use_predicted_caption=true \
 model.use_attention=true \
 engine.n_epochs=800 \
 engine.batchwise_train=false \
 engine.eval_only=false \
 resume=False \
 loss.weighting_type='adaptative' \
 optimizer.lr=0.01 \
 batch_size=128 \
 debug=false \