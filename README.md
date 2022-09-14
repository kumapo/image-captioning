# Image captioning with vision-encoder-decoder

Image captioning with huggingface's VisionEncoderDecoderModel.

## Train

`python -m image_captioning.train`

## Evaluate

`python -m image_captioning.evaluate`

## Installation 

- clone it with git
- `pipenv install`

## Results

### [kumapo/swin-gpt2-ja-image-captioning](https://huggingface.co/kumapo/swin-gpt2-ja-image-captioning)

```
Validation metrics: {'eval_loss': 1.958005666732788, 
'eval_bleu': 0.0, 'eval_precisions': [0.002952430324125503, 0.0, 0.0, 0.0], 'eval_brevity_penalty': 0.9956204701428368, 'eval_length_ratio': 0.9956300323446425, 'eval_translation_length': 202545, 'eval_reference_length': 203434, 
'eval_rouge1': 0.008349651533534636, 'eval_rouge2': 9.71097504773191e-05, 'eval_rougeL': 0.008327631368565637, 'eval_rougeLsum': 0.008334403186516591, 
'eval_meteor': 0.0014763973928500888, 'eval_runtime': 28902.4232, 'eval_samples_per_second': 7.007, 'eval_steps_per_second': 0.219}
```

### [kumapo/vit-gpt2-ja-image-captioning](https://huggingface.co/kumapo/vit-gpt2-ja-image-captioning)

```
Validation metrics: {'eval_loss': 4.004899501800537, 
'eval_bleu': 0.0, 'eval_precisions': [0.016446505117662498, 0.0011479738261967627, 0.0012748597654258032, 0.0], 'eval_brevity_penalty': 0.46042275320713894, 'eval_length_ratio': 0.5631866778341627, 'eval_translation_length': 34293, 'eval_reference_length': 60891, 
'eval_rouge1': 0.0, 'eval_rouge2': 0.0, 'eval_rougeL': 0.0, 'eval_rougeLsum': 0.0, 
'eval_meteor': 0.003167212891240735, 'eval_runtime': 8232.0141, 'eval_samples_per_second': 3.108, 'eval_steps_per_second': 0.097}
```

### swin-transformer and bert-base-japanese-v2

```
Validation metrics: {'eval_loss': 1.800550103187561, 
'eval_bleu': 0.037993467956122476, 'eval_precisions': [0.16161329932936622, 0.060897745961110576, 0.022454644592049173, 0.009428687608010572], 'eval_brevity_penalty': 1.0, 'eval_length_ratio': 3.6875653082549635, 'eval_translation_length': 42348, 'eval_reference_length': 11484, 
'eval_rouge1': 0.006416666666666667, 'eval_rouge2': 0.0, 'eval_rougeL': 0.006166666666666667, 'eval_rougeLsum': 0.006333333333333334, 
'eval_meteor': 0.321189422057437, 'eval_runtime': 57.7106, 'eval_samples_per_second': 17.328, 'eval_steps_per_second': 0.554}
```

### swin-transformer and bert-base

```
Validation metrics: {'eval_loss': 1.8393419981002808, 
'eval_bleu': 0.09459674644216093, 'eval_precisions': [0.3914987919173887, 0.13178894294736912, 0.0556269031931807, 0.02790040543839778], 'eval_brevity_penalty': 1.0, 'eval_length_ratio': 1.1499486780164938, 'eval_translation_length': 324895, 'eval_reference_length': 282530, 
'eval_rouge1': 0.3889209673113996, 'eval_rouge2': 0.14290209841020052, 'eval_rougeL': 0.34898281137141984, 'eval_rougeLsum': 0.34897135758381004, 
'eval_meteor': 0.3299746447586535, 'eval_runtime': 1545.103, 'eval_samples_per_second': 16.189, 'eval_steps_per_second': 0.506}

```

### [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

```
Validation metrics: {'eval_loss': 3.848054885864258, 
'eval_bleu': 0.07844469436156895, 'eval_precisions': [0.3756299519278358, 0.13535894282476066, 0.05933713959776451, 0.029570866467069676], 'eval_brevity_penalty': 0.8071497542053976, 'eval_length_ratio': 0.8235563069529065, 'eval_translation_length': 232359, 'eval_reference_length': 282141, 
'eval_rouge1': 0.40622458302154774, 'eval_rouge2': 0.15348220560040615, 'eval_rougeL': 0.37008334473957355, 'eval_rougeLsum': 0.37008484311066436, 
'eval_meteor': 0.28429519416686094, 'eval_runtime': 2809.3776, 'eval_samples_per_second': 8.904, 'eval_steps_per_second': 0.278}
```
