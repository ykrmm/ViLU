python lscaleuq/run.py \
 wandbconf.name="Test_ConfidNet_ImageNet" \
 datasets=ImageNetEmb \
 models=ConfidNet \
 backbone_model.name=ViT-B/32 \
 losses=Weighted_BCE \
 gpu=0 \
 engine.lr=0.01 \
 engine.n_epochs=50 \
 engine.eval_freq=1 \
 engine.batch_size=512 \
 engine.eval_only=True \
 engine.template=imagenet_base \