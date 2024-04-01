<div align="center">

# StableTTS

Next-generation TTS model using flow-matching and DiT, inspired by [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3).


</div>

## Introduction

StableTTS is fast and lightweight TTS model for chinese and english speech generation. It has only 10M parameters

**Work is in progress now. Pretrained models and detailed instructions will be released soon!**


## Inference

For detailed inference instructions, please refer to `inference.ipynb`

## Training

Setting up and training your model with StableTTS is straightforward. Follow these steps to get started:

### Preparing Your Data

1. **Generate Text and Audio pairs**: Please generate the text and audio pair filelist as `./filelists/example.txt`. Some recipes of open-source datasets could be found in `./recipes`.

2. **Run Preprocessing**: Run `preprocess.py`. This script will process the audios and texts based on the filelist.

### Start training

1. **Adjust Training Configuration**: Edit `TrainConfig` in `config.py` to specify the file list path and tweak training hyperparameters to your needs.

2. **Start the Training Process**: Launch `train.py` to begin training your model.

### Experiment with Configurations

Feel free to explore and modify settings in `config.py` to modify the hyperparameters of vocos!

## References

Most of the codes are borrowed from:

[Matcha TTS](https://github.com/shivammehta25/Matcha-TTS)

[Grad TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)