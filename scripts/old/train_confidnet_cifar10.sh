python lscaleuq/run.py \
 wandbconf.name="Repro_ConfidNet_CIFAR10" \
 datasets=CIFAR10 \
 models=ConfidNet \
 models.model.activation=relu \
 models.model.layers=[512,512,256,128,1] \
 backbone_model=CLIP_ViT_B32 \
 losses=BCE \
 losses.loss.weight=10.0 \
 optim.optimizer.lr=0.01 \
 engine.n_epochs=300 \
 engine.eval_freq=1 \