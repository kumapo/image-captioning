import argparse
import transformers
import torch
import datasets
import evaluate
import PIL
import os
import pathlib

import numpy as np

from tqdm.notebook import tqdm
from .utils import (
    seed_everything,
    save_args
)

def main(args: argparse.Namespace):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print('train: args=%s' % args.__dict__)
    save_args(args, args.output_dir / 'train_args.json')

    feature_extractor = transformers.DeiTFeatureExtractor.from_pretrained(args.encoder_model_name_or_path)
    tokenizer = transformers.ElectraTokenizer.from_pretrained(args.decoder_model_name_or_path)
    model = transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_model_name_or_path, args.decoder_model_name_or_path
    )
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = datasets.load_dataset(
        "kumapo/coco_dataset_script", "2017", 
        data_dir=str(args.train_data_dir), split="train", streaming=True
    )
    eval_dataset = datasets.load_dataset(
        "kumapo/coco_dataset_script", "2017",
        data_dir=str(args.valid_data_dir), split="validation", streaming=True
    )
    # https://github.com/huggingface/datasets/issues/4675
    def preprocess_function(examples):
        # prepare image (i.e. resize + normalize)
        pixel_values = feature_extractor(
            [PIL.Image.open(path).convert("RGB") for path in examples['image_path']],
            return_tensors="np"
        ).pixel_values
        # add labels (input_ids) by encoding the text
        encoded = tokenizer(
            [label for label in examples['caption']], 
            padding="max_length",
            max_length=args.max_sequence_length,
            return_tensors="np",
            return_length=True
        )
        del examples
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": encoded.input_ids,
            "length": encoded.length
        }

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["image_path","caption"]
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["image_path","caption"]
    )
    train_dataloader = torch.utils.data.DataLoader(
        # https://github.com/huggingface/datasets/discussions/2577
        train_dataset.shuffle(seed=args.random_seed, buffer_size=1000).take(args.num_train_data).with_format("torch"),
        # take and skip prevent future calls to shuffle because they lock in the order of the shards. 
        # You should shuffle your dataset before splitting it.
        batch_size=args.train_batch_size,
        num_workers=args.num_workers # must be 1 otherwise a thread crashs
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset.take(args.num_valid_data).with_format("torch"),
        batch_size=args.valid_batch_size,
        num_workers=args.num_workers # above
    )

    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate)
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    for epoch in range(args.num_train_epochs):  # loop over the dataset multiple times
        # train
        model.train()
        train_losses = []
        for batch in tqdm(train_dataloader):
            # important: make sure that PAD tokens are ignored by the loss function
            batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100
            # get the inputs        
            batch = {
                k: v.to(device)
                for k,v in batch.items() if k in ("pixel_values","labels")
            }
            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del batch

        train_losses.append(loss.item())
        print(f"Loss after epoch {epoch}:", sum(train_losses)/len(train_losses))

        # evaluate
        model.eval()
        outputs = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                # run batch generation
                batch_output = model.generate(batch["pixel_values"].to(device))
                outputs += tokenizer.batch_decode(
                    batch_output.cpu().numpy(),
                    skip_special_tokens=True
                )
                labels += tokenizer.batch_decode(
                    batch["labels"],
                    skip_special_tokens=True
                )
                del batch

        # compute metrics
        metrics = {}
        try:
            metrics.update(bleu.compute(predictions=outputs, references=labels))
        except ZeroDivisionError as e:
            metrics.update(dict(bleu="nan"))
        try:
            metrics.update(meteor.compute(predictions=outputs, references=labels))
        except ZeroDivisionError as e:
            metrics.update(dict(meteor="nan"))
        print(f"Train metrics after {epoch}:", metrics)

    # evaluate
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # run batch generation
            batch_output = model.generate(batch["pixel_values"].to(device))
            outputs += tokenizer.batch_decode(
                batch_output.cpu().numpy(),
                skip_special_tokens=True
            )
            labels += tokenizer.batch_decode(
                batch["labels"],
                skip_special_tokens=True
            )
            del batch

    # compute metrics
    metrics = {}
    try:
        metrics.update(bleu.compute(predictions=outputs, references=labels))
    except ZeroDivisionError as e:
        metrics.update(dict(bleu="nan"))
    try:
        metrics.update(meteor.compute(predictions=outputs, references=labels))
    except ZeroDivisionError as e:
        metrics.update(dict(meteor="nan"))
    print(f"Validation metrics:", metrics)

    # save finally
    if not os.path.exists('output/'):
        os.mkdir('output/')
    model.save_pretrained('output/')
 
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_dir", default='../input/coco-2017-train/', type=pathlib.Path
    )
    parser.add_argument(
        "--valid_data_dir", default='../input/coco-2017-val/', type=pathlib.Path
    )
    parser.add_argument(
        "--output_dir", default=pathlib.Path('output'), type=pathlib.Path, help=""
    )
    parser.add_argument(
        "--encoder_model_name_or_path", default="facebook/deit-tiny-patch16-224", type=str, help=""
    )
    parser.add_argument(
        "--decoder_model_name_or_path", default="google/electra-small-discriminator", type=str, help=""
    )
    parser.add_argument(
        "--max_sequence_length", default=64, type=int, help=""
    )
    parser.add_argument(
        "--num_train_epochs", default=2, type=int, help=""
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help=""
    )
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help=""
    )
    parser.add_argument(
        "--valid_batch_size", default=32, type=int, help=""
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help=""
    )
    parser.add_argument(
        "--random_seed", default=42, type=int, help="Random seed for determinism."
    )
    parser.add_argument(
        "--num_train_data", default=10000, type=int, help="number of items to train on dataset."
    )
    parser.add_argument(
        "--num_valid_data", default=2000, type=int, help="number of items to evaluate on dataset."
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    args = parser.parse_args()
    seed_everything(args.random_seed)
    main(args)
