python -m image_captioning.train \
   --encoder_model_name_or_path google/vit-base-patch16-224 \
   --train_dataset_config_names sp000 \
   --valid_dataset_config_names sp000 \
   --num_train_data 32 \
   --num_valid_data 32 \
   --train_data_dir input/ \
   --valid_data_dir input/ \
   --no_fp16