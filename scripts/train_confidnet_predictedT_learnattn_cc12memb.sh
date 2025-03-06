PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES="0" python lscaleuq/run.py \
 --multirun \
 exp_name="learnatnn_predicted_text" \
 dataset=cc12m_emb \
 backbone=clip_vit_b32 \
 zs_datasets=data_suite+imgnet \
 model=confidnetvlm_attention \
 model.keep_frozen=false \
 model.n_iter_freeze_proj=312500,468750,781250,1562500 \
 model.use_predicted_caption=true \
 engine.n_epochs=400 \
 engine.batchwise_train=true \
 engine.eval_zs=true \
 engine.eval_only=false \
 resume=False \
 loss.weight=1.5 \
 optimizer.lr=0.001 \
 batch_size=512 \
 cluster_env=cnam \
 wandb_mode=online
