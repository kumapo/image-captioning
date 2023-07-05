# Image captioning with vision-encoder-decoder

Image captioning with VisionEncoderDecoderModel from huggingface.

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
Validation metrics: {'eval_loss': 1.9995830059051514, 'eval_score': 14.845372889809905, 'eval_counts': [5353, 2245, 954, 477], 'eval_totals': [10740, 9740, 8740, 7740], 'eval_precisions': [49.84171322160149, 23.04928131416838, 10.915331807780321, 6.162790697674419], 'eval_bp': 0.8903790505341749, 'eval_sys_len': 10740, 'eval_ref_len': 11987, 'eval_rouge1': 0.37680551175187166, 'eval_rouge2': 0.14616265481773555, 'eval_rougeL': 0.350276990789434, 'eval_rougeLsum': 0.350436066526591, 'eval_meteor': 0.0005, 'eval_runtime': 124.1608, 'eval_samples_per_second': 8.054, 'eval_steps_per_second': 0.258}
```

### [kumapo/vit-gpt2-ja-image-captioning](https://huggingface.co/kumapo/vit-gpt2-ja-image-captioning)

```
Validation metrics: {'eval_loss': 2.059309720993042, 'eval_score': 14.541195244037787, 'eval_counts': [5289, 2216, 925, 464], 'eval_totals': [10709, 9709, 8709, 7709], 'eval_precisions': [49.38836492669717, 22.82418374703883, 10.621196463428637, 6.018938902581398], 'eval_bp': 0.8875069968895519, 'eval_sys_len': 10709, 'eval_ref_len': 11987, 'eval_rouge1': 0.3721238135778884, 'eval_rouge2': 0.1433057833911217, 'eval_rougeL': 0.3486764739133251, 'eval_rougeLsum': 0.3488526198999652, 'eval_meteor': 0.0, 'eval_runtime': 293.0325, 'eval_samples_per_second': 3.413, 'eval_steps_per_second': 0.109}
```

### swin-transformer and bert-base-japanese-v2

```
Validation metrics: {'eval_loss': 1.6615628004074097, 'eval_score': 15.270138656859801, 'eval_counts': [6013, 2574, 1113, 542], 'eval_totals': [13002, 12002, 11002, 10002], 'eval_precisions': [46.24673127211198, 21.446425595734045, 10.116342483184875, 5.4189162167566485], 'eval_bp': 1.0, 'eval_sys_len': 13002, 'eval_ref_len': 12193, 'eval_rouge1': 0.46499010155295006, 'eval_rouge2': 0.2229896169710825, 'eval_rougeL': 0.40137460532380376, 'eval_rougeLsum': 0.4018381882765504, 'eval_meteor': 0.40398952495284224, 'eval_runtime': 54.0277, 'eval_samples_per_second': 18.509, 'eval_steps_per_second': 0.592}
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
