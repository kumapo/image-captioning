"""
{commit_data}
"""
import gzip
import base64
import os
import pathlib
import typing

# this is base64 encoded source code
file_data: typing.Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


!cp /opt/conda/lib/libstdc++.so.6.0.30 /lib/x86_64-linux-gnu/
!rm -rf /lib/x86_64-linux-gnu/libstdc++.so.6
!ln -s /lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /lib/x86_64-linux-gnu/libstdc++.so.6
!pip install -q git+https://github.com/huggingface/transformers.git@main evaluate datasets==2.4.0 sacrebleu[ja] rouge_score fugashi unidic-lite

# !python -m image_captioning.preprocess
#    --encoder_model_name_or_path "microsoft/swin-base-patch4-window7-224-in22k" \
#    --decoder_model_name_or_path "rinna/japanese-gpt2-medium"

# !python -m image_captioning.train \
#    --encoder_model_name_or_path "google/vit-base-patch16-224" \
#    --decoder_model_name_or_path "rinna/japanese-gpt2-medium" \
#     --num_train_data 80000 \
#     --num_valid_data 1000 \
#     --train_batch_size 16 \
#     --eval_steps 1200

!python -m image_captioning.evaluate \
   --encoder_decoder_model_name_or_path ../input/image-captioning-v224/output \
   --num_test_data 32 \
   --num_workers 0
