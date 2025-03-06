python lscaleuq/run.py \
 wandbconf.name="Gen_Emb_ImageNet" \
 datasets=ImageNet \
 models=ConfidNet \
 engine=Generate_emb \
 backbone_model.name=ViT-B/32 \
 losses=Weighted_BCE \
 gpu=0 \