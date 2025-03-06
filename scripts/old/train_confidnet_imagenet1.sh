python lscaleuq/run.py \
 --multirun \
 wandbconf.name="Test_ConfidNet_AugmentationImageNet" \
 datasets=ImageNet \
 models=ConfidNet \
 backbone_model.name=ViT-B/32 \
 losses=MSE,BCE \
 losses.weighting=10.0 \
 gpu=1 \
 engine.lr=0.001 \
 optim=SGD \
 engine.n_epochs=50 \
 engine.eval_freq=1 \
 engine.batch_size=512 \
 engine.template=imagenet_base \