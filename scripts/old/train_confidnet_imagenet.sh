python lscaleuq/run.py \
 --multirun \
 wandbconf.name="Test_ConfidNet_AugmentationImageNet" \
 datasets=ImageNet \
 models=ConfidNet \
 backbone_model.name=ViT-B/32 \
 losses=BCE,MSE \
 losses.weighting=10.0 \
 gpu=0 \
 engine.lr=0.01 \
 optim=SGD \
 engine.n_epochs=50 \
 engine.eval_freq=1 \
 engine.batch_size=512 \
 engine.template=imagenet_base \