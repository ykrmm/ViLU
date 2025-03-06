PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES="1" python lscaleuq/run.py \
 exp_name="predicted_text_only" \
 dataset=imagenet_emb \
 backbone=clip_vit_b32 \
 model=confidnetvlm_attention \
 model.keep_frozen=true \
 model.n_iter_freeze_proj=1000 \
 model.use_predicted_caption=true \
 model.use_attention=false \
 engine.n_epochs=250 \
 engine.eval_zs=true \
 resume=False \
 loss.weight=1.5 \
 optimizer.lr=0.001 \
 batch_size=512 \
 cluster_env=cnam \
 wandb_mode=online
