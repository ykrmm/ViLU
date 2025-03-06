python lscaleuq/run.py \
 exp_name=ConfidNetVLM_Wopt_ImageNet \
 dataset=imagenet \
 model=confidnetvlm_attention \
 backbone=clip_vit_b32 \
 loss=bce \
 loss.weight=1.5 \
 optimizer=sgd \
 optimizer.lr=0.001 \
 engine.n_epochs=150 \
 batch_size=512 \
 engine.eval_only=False \
 engine.eval_freq=1 \
 engine.template=base \