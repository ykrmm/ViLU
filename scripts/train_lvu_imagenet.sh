PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES="0" python vilu/run.py \
 exp_name="lvu_imgnet" \
 dataset=imagenet \
 backbone=clip_vit_b32 \
 model=vilu_attention \
 model.keep_frozen=true \
 model.n_iter_freeze_proj=1000 \
 model.use_predicted_caption=false \
 model.use_attention=false \
 engine.n_epochs=300 \
 engine.batchwise_train=false \
 engine.eval_only=false \
 resume=False \
 loss=mse \
 loss.weighting_type='adaptative' \
 optimizer.lr=0.001 \
 batch_size=512 \
 debug=false
