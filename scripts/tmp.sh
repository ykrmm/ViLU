CUDA_VISIBLE_DEVICES="0,1,2" python lscaleuq/run.py \
 exp_name="predicted_text_v2" \
 dataset=cifar10 \
 backbone=clip_vit_b32 \
 model=confidnetvlm_attention \
 model.keep_frozen=false \
 model.n_iter_freeze_proj=1000 \
 model.use_predicted_caption=true \
 model.layers=[512,256,128,1] \
 engine.n_epochs=800 \
 resume=False \
 loss.weight=10.0 \
 optimizer.lr=0.01 \
 batch_size=128 \
 cluster_env=local_cnam \
 wandb_mode=online \




 CUDA_VISIBLE_DEVICES="0,1,2" python lscaleuq/run.py \
 exp_name=pet_food_flowers_debug \
 model=confidnetvlm_attention \
 dataset=pets_food_flowers_fix \
 shuffle_val=true \
 optimizer.lr=1e-2 \
 backbone=clip_vit_b32 \
 engine.n_epochs=100 \
 engine.batchwise_train=true \
 engine.eval_only=false \
 engine.print_freq=5 \
 engine.display_progress=true \
 engine.template_type=imagenet_base \
 engine.eval_zs=true \
 engine.eval_zs_freq=10 \
 engine.eval_freq=10 \
 zs_datasets=pets_food_flowers \
 loss.weight=1.5 \
 model.keep_frozen=false \
 model.n_iter_freeze_proj=1000 \
 resume=false \
 batch_size=512 \
 batch_size_val=512 \
 cluster_env=local_cnam \
 wandb_mode=online \