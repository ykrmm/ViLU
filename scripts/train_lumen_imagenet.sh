PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES="0" python vilu/run.py \
 exp_name="vilu_imgnet" \
 dataset=imagenet \
 backbone=clip_vit_b32 \
 model=vilu_attention \
 model.n_iter_freeze_proj=1000 \
 model.use_predicted_caption=true \
 model.use_attention=true \
 engine.n_epochs=300 \
 engine.batchwise_train=false \
 engine.eval_only=false \
 resume=False \
 loss.weighting_type='adaptative' \
 optimizer.lr=0.001 \
 batch_size=512 \
 debug=false \