import argparse
import transformers
import torch
import datasets
import evaluate
import PIL
import os
import pathlib

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from .utils import (
    seed_everything,
    save_args
)


def main(args: argparse.Namespace):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print('preprocess: args=%s' % args.__dict__)
    save_args(args, args.output_dir / 'preprocess_args.json')

    model = transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_model_name_or_path, args.decoder_model_name_or_path
    )
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(args.encoder_model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.decoder_model_name_or_path)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = datasets.load_dataset(
        "kumapo/stair_captions_dataset_script", "2014", 
        data_dir=str(args.train_data_dir), split="train", streaming=True
    )
    # https://github.com/huggingface/datasets/issues/4675
    def preprocess_function(examples):
        # add labels (input_ids) by encoding the text
        encoded = tokenizer(
            [label for label in examples['caption']], 
            padding="do_not_pad",
            max_length=args.max_sequence_length,
            return_tensors="np",
            return_length=True
        )
        del examples
        # important: make sure that PAD tokens are ignored by the loss function
        length = encoded.input_ids[encoded.input_ids != tokenizer.pad_token_id].sum()
        return {
            # "pixel_values": pixel_values.squeeze(),
            # "labels": encoded.input_ids,
            "length": encoded.length
        }

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["image_path", "caption", 'image_id', 'width', 'file_name', 'coco_url', 'caption_id', 'height']
    )
    train_dataset = datasets.Dataset.from_dict(
        train_dataset._head(args.num_train_data),
        features=datasets.Features({
            # "labels": datasets.Sequence(feature=datasets.Value(dtype='int64'), length=args.max_sequence_length),
            "length": datasets.Value(dtype='int32')
        })
    ).with_format("torch")
    train_df = train_dataset = train_dataset.to_pandas()
    train_df.to_csv(args.output_dir / 'preprocessed_train.csv')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_dir", default='../input/coco-2014-train/', type=pathlib.Path
    )
    parser.add_argument(
        "--valid_data_dir", default='../input/coco-2014-val/', type=pathlib.Path
    )
    parser.add_argument(
        "--output_dir", default=pathlib.Path('output'), type=pathlib.Path, help=""
    )
    parser.add_argument(
        "--encoder_model_name_or_path", default="microsoft/swin-base-patch4-window7-224-in22k", type=str, help=""
    )
    parser.add_argument(
        "--decoder_model_name_or_path", default="cl-tohoku/bert-base-japanese-v2", type=str, help=""
    )
    parser.add_argument(
        "--max_sequence_length", default=64, type=int, help=""
    )
    parser.add_argument(
        "--random_seed", default=42, type=int, help="Random seed for determinism."
    )
    parser.add_argument(
        "--num_train_data", default=82783, type=int, help="number of items to train on dataset."
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    args = parser.parse_args()
    seed_everything(args.random_seed)
    main(args)
