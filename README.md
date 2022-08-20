# Image captioning with vision-encoder-decoder

An example of image captioning with huggingface's VisionEncoderDecoderModel.

## Train

`python -m image_captioning.train`

## Evaluate

`python -m image_captioning.evaluate`

## Installation 

- clone it with git
- `pipenv install`

## 

Validation metrics: {'eval_loss': 1.9863734245300293, 'eval_bleu': 0.019937610789262112, 'eval_precisions': [0.12167217123150192, 0.038640967975763434, 0.010681211294812795, 0.003146535914908403], 'eval_brevity_penalty': 1.0, 'eval_length_ratio': 4.885273667320378, 'eval_translation_length': 54803, 'eval_reference_length': 11218, 'eval_meteor': 0.2621822878759513, 'eval_runtime': 55.8558, 'eval_samples_per_second': 17.903, 'eval_steps_per_second': 0.573}


Validation metrics: {'eval_loss': 2.1743035316467285, 'eval_bleu': 0.01733187360427669, 'eval_precisions': [0.11765127360011464, 0.03500164155692555, 0.008973358599933118, 0.0024419793283610345], 'eval_brevity_penalty': 1.0, 'eval_length_ratio': 4.976466393296488, 'eval_translation_length': 55826, 'eval_reference_length': 11218, 'eval_meteor': 0.258356268559834, 'eval_runtime': 55.1899, 'eval_samples_per_second': 18.119, 'eval_steps_per_second': 0.58}
