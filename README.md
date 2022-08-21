# Image captioning with vision-encoder-decoder

An example of image captioning with huggingface's VisionEncoderDecoderModel.

## Train

`python -m image_captioning.train`

## Evaluate

`python -m image_captioning.evaluate`

## Installation 

- clone it with git
- `pipenv install`

## Results

### swin-transformer and bert-base

```
Validation metrics: {'eval_loss': 1.8393419981002808, 
'eval_bleu': 0.023860199318381667, 'eval_precisions': [0.13233766058244154, 0.043730314585396675, 0.013333995686425487, 0.004200199357288335], 'eval_brevity_penalty': 1.0, 'eval_length_ratio': 4.4520688068523695, 'eval_translation_length': 1257843, 'eval_reference_length': 282530, 
'eval_rouge1': 0.20548773633832365, 'eval_rouge2': 0.07244194056078415, 'eval_rougeL': 0.1771188682107597, 'eval_rougeLsum': 0.177116422490797, 'eval_meteor': 0.27636459119680745, 'eval_runtime': 1844.2857, 'eval_samples_per_second': 13.563, 'eval_steps_per_second': 0.424}
```

### "nlpconnect/vit-gpt2-image-captioning"

```
Validation metrics: {'eval_loss': 3.848054885864258, 
'eval_bleu': 0.04315136377668535, 'eval_precisions': [0.3056662039964848, 0.08420594965675057, 0.0272629544256432, 0.009319055985564366], 'eval_brevity_penalty': 0.8533177231692343, 'eval_length_ratio': 0.8630932760570069, 'eval_translation_length': 243514, 'eval_reference_length': 282141, 
'eval_rouge1': 0.33861043618402087, 'eval_rouge2': 0.0992569266494081, 'eval_rougeL': 0.3036297776304849, 'eval_rougeLsum': 0.3035955315381848, 
'eval_meteor': 0.2286512157841887, 'eval_runtime': 1514.1369, 'eval_samples_per_second': 16.52, 'eval_steps_per_second': 0.516}
```
