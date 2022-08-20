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
        args.preprocessor_name_or_path if args.preprocessor_name_or_path is not None else args.encoder_decoder_model_name_or_path
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.encoder_decoder_model_name_or_path)
    # tokenizer = transformers.GPT2TokenizerFast.from_pretrained(args.encoder_decoder_model_name_or_path)
    # feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(args.encoder_decoder_model_name_or_path)

    test_dataset = datasets.load_dataset(
        "kumapo/coco_dataset_script", "2017",
        data_dir=str(args.test_data_dir), split=args.test_data_split, streaming=True
    )

    # https://github.com/huggingface/datasets/issues/4675
    def preprocess_function(batch):
        # prepare image (i.e. resize + normalize)
        pixel_values = feature_extractor(
            [PIL.Image.open(path).convert("RGB") for path in batch['image_path']],
            return_tensors="np"
        ).pixel_values
        # add labels (input_ids) by encoding the text
        encoded = tokenizer(
            [label for label in batch['caption']], 
            padding="max_length",
            max_length=args.max_sequence_length,
            return_tensors="np",
            return_length=True
        )
        del batch
        # important: make sure that PAD tokens are ignored by the loss function
        encoded.input_ids[encoded.input_ids == tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": encoded.input_ids,
            # "length": encoded.length
        }

    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["image_path","caption_id","caption","coco_url","file_name","height","width"]
    )
    # eval_dataloader = torch.utils.data.DataLoader(
    #     test_dataset.take(args.num_test_data).with_format("torch"),
    #     batch_size=args.test_batch_size,
    #     num_workers=args.num_workers # above
    # )

    if 0 < args.num_test_data:
        test_dataset = datasets.Dataset.from_dict(
            test_dataset._head(args.num_test_data),
            features=datasets.Features({
                "pixel_values": datasets.Array3D(shape=(3, 224, 224), dtype='float32'),
                "labels": datasets.Sequence(feature=datasets.Value(dtype='int32'), length=args.max_sequence_length)
            })
        ).with_format("torch")
    else:
        test_dataset = test_dataset.with_format("torch")

    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
    training_args = transformers.Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_eval_batch_size=args.test_batch_size,
        fp16=False if args.debug else not args.no_fp16, 
        output_dir=args.output_dir,
        dataloader_num_workers=args.num_workers,
        report_to="tensorboard",
        seed=args.random_seed
    )

    # bleu_metric = datasets.load_metric("sacrebleu")
    # meteor_metric = datasets.load_metric("meteor")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        metrics = {}
        try:
            metrics.update(bleu_metric.compute(predictions=pred_str, references=label_str))
        except ZeroDivisionError as e:
            metrics.update(dict(bleu="nan"))
        try:
            metrics.update(meteor_metric.compute(predictions=pred_str, references=label_str))
        except ZeroDivisionError as e:
            metrics.update(dict(meteor="nan"))
        return metrics
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # instantiate trainer
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=test_dataset,
        data_collator=transformers.default_data_collator,
    )

    # https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/generation_utils.py#L845
    gen_kwargs = dict(
        do_sample=True, 
        max_length=args.max_sequence_length, 
        top_k=50, 
        top_p=0.9, 
        num_return_sequences=1
    )
    # evaluate
    metrics = trainer.evaluate(test_dataset, **gen_kwargs)
    print("Validation metrics:", metrics)

    def forward_pass_with_label(batch):
        #  Creating a tensor from a list of numpy.ndarrays is extremely slow.
        inputs = {
            k:torch.tensor(np.array(v)).to(device) for k,v in batch.items()
            if k in ("pixel_values", "labels")
        }
        generated_ids = model.generate(
            inputs["pixel_values"].to(device),
            **gen_kwargs,
        )
        generated_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        label_ids = np.array(batch["labels"])
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        with torch.no_grad():
            # calc loss manually
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(outputs.logits.reshape(-1, model.decoder.config.vocab_size), inputs["labels"].view(-1))
            loss = loss.cpu().numpy().reshape(inputs["labels"].shape[0],-1).sum(axis=1)
        del inputs
        return dict(
            loss=loss,
            predicted_labels=generated_str,
            labels=label_str
        )

    evaluation = test_dataset.map(
        forward_pass_with_label,
        batched=True,
        batch_size=args.test_batch_size,
        remove_columns=["pixel_values"],
        drop_last_batch=False
    )
    evaluation = evaluation._head(
        args.num_test_data if 0 < args.num_test_data else 5000
    )
    eval_df = pd.DataFrame(evaluation)
    eval_df.to_csv(args.output_dir / "evaluation.csv")

    # prediction
    # pred = trainer.predict(test_dataset.take(3).with_format("torch"), **gen_kwargs)
    # labels_ids = pred.label_ids
    # pred_ids = pred.predictions
    # pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # pred_str = [pred for pred in pred_str]
    # labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    # label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # print("Validation predictions:", pred_str)
    # print("Validation labels:", label_str)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_dir", default='../input/coco-2017-val/', type=pathlib.Path
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
        "--preprocessor_name_or_path", default=None, type=str, help=""
    )
    parser.add_argument(
        "--max_sequence_length", default=64, type=int, help=""
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
