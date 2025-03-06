python lscaleuq/run.py \
 --multirun \
 wandbconf.name="Search_ConfidNetv2_ImageNet" \
 datasets=ImageNetEmb \
 models=ConfidNet \
 models.train_confidnet.activation=leaky_relu,relu \
 models.train_confidnet.layers=[512,512,256,128,1],[512,512,512,1] \
 backbone_model.name=ViT-B/32 \
 losses=BCE \
 losses.weighting=10.0 \
 optim=SGD \
 gpu=2 \
 engine.lr=0.001 \
 engine.n_epochs=50 \
 engine.batch_size=512 \
 engine.eval_freq=1 \
 engine.template=imagenet_base \