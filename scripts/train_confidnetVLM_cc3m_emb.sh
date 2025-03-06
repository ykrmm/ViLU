python lscaleuq/run.py \
 exp_name=ConfidNetVLM_Wopt_cc3m_emb \
 dataset=cc3m_emb \
 model=confidnetvlm_attention \
 backbone=clip_vit_b32 \
 loss=bce \
 loss.weight=1.5 \
 optimizer=sgd \
 optimizer.lr=0.001 \
 batch_size=512 \
 engine.n_epochs=150 \
 engine.eval_only=False \
 engine.eval_freq=1 \
 engine.batchwise_train=True \
 cluster_env=local_cnam \
 wandb_mode=disabled