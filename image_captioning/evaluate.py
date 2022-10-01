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
    print('evaluate: args=%s' % args.__dict__)
    save_args(args, args.output_dir / 'evaluate_args.json')

    # load a fine-tuned image captioning model and corresponding tokenizer and feature extractor
    model = transformers.VisionEncoderDecoderModel.from_pretrained(args.encoder_decoder_model_name_or_path)
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
        args.feature_extractor_name_or_path if args.feature_extractor_name_or_path is not None else args.encoder_decoder_model_name_or_path
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.encoder_decoder_model_name_or_path
    )
    # tokenizer = transformers.GPT2TokenizerFast.from_pretrained(args.encoder_decoder_model_name_or_path)
    # feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(args.encoder_decoder_model_name_or_path)

    test_dataset = datasets.load_dataset(
        "kumapo/stair_captions_dataset_script", "2014",
        data_dir=str(args.test_data_dir), split=args.test_data_split, streaming=True
    )
    # https://github.com/huggingface/datasets/issues/4675
    def preprocess_function(examples):
        do_padding = False if tokenizer.pad_token_id is None else True
        # prepare image (i.e. resize + normalize)
        pixel_values = feature_extractor(
            [PIL.Image.open(path).convert("RGB") for path in examples['image_path']],
            return_tensors="np"
        ).pixel_values
        # add labels (input_ids) by encoding the text
        encoded = tokenizer(
            [label for label in examples['caption']], 
            padding="max_length" if do_padding else "do_not_pad",
            max_length=args.max_sequence_length,
            truncation=True,
            return_tensors="np"
        )
        image_ids = examples["image_id"]
        del examples
        if do_padding:
            # important: make sure that PAD tokens are ignored by the loss function
            encoded.input_ids[encoded.input_ids == tokenizer.pad_token_id] = -100
        else:
            encoded.input_ids = [
                input_ids + ([-100] * (args.max_sequence_length - len(input_ids)))
                for input_ids in encoded.input_ids
            ]
        return {
            "id": image_ids,
            "pixel_values": pixel_values.squeeze(),
            "labels": encoded.input_ids
        }

    removed_columns = ["image_id","image_path","caption_id","caption","coco_url","file_name","height","width"]
    if 0 < args.num_test_data:
        test_dataset = datasets.Dataset.from_dict(
            test_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=removed_columns
            )._head(args.num_test_data),
            features=datasets.Features({
                "id": datasets.Value(dtype='int64'),
                "pixel_values": datasets.Array3D(shape=(3, 224, 224), dtype='float32'),
                "labels": datasets.Sequence(feature=datasets.Value(dtype='int32'), length=args.max_sequence_length)
            })
        ).with_format("torch")
    else:
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=removed_columns
        ).with_format("torch")

    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
    training_args = transformers.Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_eval_batch_size=args.test_batch_size,
        fp16=False if args.debug else not args.no_fp16, 
        output_dir=args.output_dir,
        dataloader_num_workers=args.num_workers if not args.debug else 0,
        report_to="tensorboard",
        seed=args.random_seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # def compute_metrics(pred):
    #     pred_ids = pred.predictions
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_ids = pred.label_ids
    #     if tokenizer.pad_token_id is not None:
    #         label_ids[label_ids == -100] = tokenizer.pad_token_id
    #     else:
    #         # special tokens are skipped
    #         label_ids[label_ids == -100] = tokenizer.eos_token_id
    #     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    #     metrics = {}
    #     try:
    #         metrics.update(bleu.compute(predictions=pred_str, references=label_str), smooth_method="floor", smooth_value=0.1)
    #         metrics.update(rouge.compute(predictions=pred_str, references=label_str))
    #         metrics.update(meteor.compute(predictions=pred_str, references=label_str))
    #     except ZeroDivisionError as e:
    #         pass
    #     return metrics
    # 
    # trainer = transformers.Seq2SeqTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     compute_metrics=compute_metrics,
    #     eval_dataset=test_dataset,
    #     data_collator=transformers.default_data_collator,
    # )
    # p140
    # and https://note.com/npaka/n/n5d296d8ae26d
    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        early_stopping=True
    )
    # evaluate
    # metrics = trainer.evaluate(**gen_kwargs)
    # print("Validation metrics:", metrics)

    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    def forward_pass_with_label(batch):
        #  Creating a tensor from a list of numpy.ndarrays is extremely slow.
        inputs = {
            k:torch.tensor(np.array(v)).to(device) for k,v in batch.items()
            if k in ("pixel_values", "labels")
        }
        pred_ids = model.generate(
            inputs["pixel_values"].to(device),
            **gen_kwargs,
        )
        # gpt2 tokenizer: https://discuss.huggingface.co/t/bpe-tokenizers-and-spaces-before-words/475
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.array(batch["labels"])
        if tokenizer.pad_token_id is not None:
            label_ids[label_ids == -100] = tokenizer.pad_token_id
        else:
            # special tokens are going to be skipped
            label_ids[label_ids == -100] = tokenizer.eos_token_id
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        bleu.add_batch(predictions=pred_str, references=label_str)
        rouge.add_batch(predictions=pred_str, references=label_str)
        meteor.add_batch(predictions=pred_str, references=label_str)
        with torch.no_grad():
            # calc loss manually
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(outputs.logits.reshape(-1, model.decoder.config.vocab_size), inputs["labels"].view(-1))
            loss = loss.cpu().numpy().reshape(inputs["labels"].shape[0],-1).sum(axis=1)
        ids = np.array(batch["id"]).astype(int) # the type would be either list or tensor
        del batch, inputs
        return dict(
            id=ids,
            loss=loss,
            predicted_labels=pred_str,
            labels=label_str
        )

    evaluation = test_dataset.map(
        forward_pass_with_label,
        batched=True,
        batch_size=args.test_batch_size,
        remove_columns=["pixel_values"],
        drop_last_batch=False
    )
    if 0 < args.num_test_data:
        # we cannot use DataFrame():
        # https://github.com/huggingface/datasets/issues/469
        eval_df = evaluation.to_pandas()
    else:
        eval_df = pd.concat([
            pd.DataFrame(_dict, index=[_dict["id"]])
            for _dict in evaluation
        ])
    print("Predicted %d data to evaluation.csv" % len(eval_df))
    eval_df.to_csv(args.output_dir / "evaluation.csv")
    
    metrics = {}
    metrics.update(bleu.compute(smooth_method="floor", smooth_value=0.1, tokenize='ja-mecab'))
    metrics.update(rouge.compute(tokenizer=lambda x: tokenizer.tokenize(x)))
    metrics.update(meteor.compute())
    print("Validation metrics:", metrics)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_dir", default='../input/coco-2014-val/', type=pathlib.Path
    )
    parser.add_argument(
        "--test_data_split", default="validation", type=str, help=""
    )
    parser.add_argument(
        "--output_dir", default=pathlib.Path('output'), type=pathlib.Path, help=""
    )
    parser.add_argument(
        "--encoder_decoder_model_name_or_path", default="nlpconnect/vit-gpt2-image-captioning", type=str, help=""
    )
    parser.add_argument(
        "--feature_extractor_name_or_path", default=None, type=str, help=""
    )
    parser.add_argument(
        "--tokenizer_name_or_path", default=None, type=str, help=""
    )
    parser.add_argument(
        "--max_sequence_length", default=64, type=int, help=""
    )
    parser.add_argument(
        "--max_new_tokens", default=16, type=int, help="which ignores the number of tokens in the prompt."
    )
    parser.add_argument(
        "--test_batch_size", default=32, type=int, help=""
    )
    parser.add_argument(
        "--num_workers", default=2, type=int, help=""
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help=""
    )
    parser.add_argument(
        "--random_seed", default=42, type=int, help="Random seed for determinism."
    )
    parser.add_argument(
        "--num_test_data", default=0, type=int, help="number of items to evaluate on dataset."
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    args = parser.parse_args()
    seed_everything(args.random_seed)
    main(args)
