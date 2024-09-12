<div align="center">

# StableTTS

Next-generation TTS model using flow-matching and DiT, inspired by [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3).


</div>

## Introduction

As the first open-source TTS model that tried to combine flow-matching and DiT, **StableTTS** is a fast and lightweight TTS model for chinese, english and japanese speech generation. It has 31M parameters. 

✨ **Huggingface demo:** [🤗](https://huggingface.co/spaces/KdaiP/StableTTS1.1)

## News

2024/10: A new autoregressive TTS model is coming soon...

2024/9: 🚀 **StableTTS V1.1 Released** ⭐ Audio quality is largely improved ⭐

⭐ **V1.1 Release Highlights:**

- Fixed critical issues that cause the audio quality being much lower than expected. (Mainly in Mel spectrogram and Attention mask)
- Introduced U-Net-like long skip connections to the DiT in the Flow-matching Decoder.
- Use cosine timestep scheduler from [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- Add support for CFG (Classifier-Free Guidance).
- Add support for [FireflyGAN vocoder](https://github.com/fishaudio/vocoder/releases/tag/1.0.0).
- Switched to [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for ODE solvers.
- Improved Chinese text frontend (partially based on [gpt-sovits2](https://github.com/RVC-Boss/GPT-SoVITS)).
- Multilingual support (Chinese, English, Japanese) in a single checkpoint.
- Increased parameters: 10M -> 31M.


## Pretrained models

### Text-To-Mel model

Download and place the model in the `./checkpoints` directory, it is ready for inference, finetuning and webui.

| Model Name | Task Details | Dataset | Download Link |
|:----------:|:------------:|:-------------:|:-------------:|
| StableTTS | text to mel | 600 hours | [🤗](https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/StableTTS/checkpoint_0.pt)|

### Mel-To-Wav model

Choose a vocoder (`vocos` or `firefly-gan` ) and place it in the `./vocoders/pretrained` directory.

| Model Name | Task Details | Dataset | Download Link |
|:----------:|:------------:|:-------------:|:-------------:|
| Vocos | mel to wav | 2k hours | [🤗](https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/vocoders/vocos.pt)|
| firefly-gan-base | mel to wav | HiFi-16kh | [download from fishaudio](https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base-generator.ckpt)|

## Installation

1. **Install pytorch**: Follow the [official PyTorch guide](https://pytorch.org/get-started/locally/) to install pytorch and torchaudio. We recommend the latest version (tested with PyTorch 2.4 and Python 3.12).

2. **Install Dependencies**: Run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

## Inference

For detailed inference instructions, please refer to `inference.ipynb`

We also provide a webui based on gradio, please refer to `webui.py`

## Training

StableTTS is designed to be trained easily. We only need text and audio pairs, without any speaker id or extra feature extraction. Here’s how to get started:

### Preparing Your Data

1. **Generate Text and Audio pairs**: Generate the text and audio pair filelist as `./filelists/example.txt`. Some recipes of open-source datasets could be found in `./recipes`.

2. **Run Preprocessing**: Adjust the `DataConfig` in `preprocess.py` to set your input and output paths, then run the script. This will process the audio and text according to your list, outputting a JSON file with paths to mel features and phonemes. 

**Note: Process multilingual data separately by changing the `language` setting in `DataConfig`**

### Start training

1. **Adjust Training Configuration**:  In `config.py`, modify `TrainConfig` to set your file list path and adjust training parameters (such as batch_size) as needed.

2. **Start the Training Process**: Launch `train.py` to start training your model. 

Note: For finetuning, download the pretrained model and place it in the `model_save_path` directory specified in  `TrainConfig`. Training script will automatically detect and load the pretrained checkpoint.

### (Optional) Vocoder training

The `./vocoder/vocos` folder contains the training and finetuning codes for vocos vocoder.

For other types of vocoders, we recommend to train by using [fishaudio vocoder](https://github.com/fishaudio/vocoder): an uniform interface for developing various vocoders. We use the same spectrogram transform so the vocoders trained is compatible with StableTTS.

## Model structure

<div align="center">

<p style="text-align: center;">
  <img src="./figures/structure.jpg" height="512"/>
</p>

</div>

- We use the Diffusion Convolution Transformer block from [Hierspeech++](https://github.com/sh-lee-prml/HierSpeechpp), which is a combination of original [DiT](https://github.com/sh-lee-prml/HierSpeechpp) and [FFT](https://arxiv.org/pdf/1905.09263.pdf)(Feed forward Transformer from fastspeech) for better prosody.

- In flow-matching decoder, we add a [FiLM layer](https://arxiv.org/abs/1709.07871) before DiT block to condition timestep embedding into model.

## References

The development of our models heavily relies on insights and code from various projects. We express our heartfelt thanks to the creators of the following:

### Direct Inspirations

[Matcha TTS](https://github.com/shivammehta25/Matcha-TTS): Essential flow-matching code.

[Grad TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS): Diffusion model structure.

[Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3): Idea of combining flow-matching and DiT.

[Vits](https://github.com/jaywalnut310/vits): Code style and MAS insights, DistributedBucketSampler.

### Additional References:

[plowtts-pytorch](https://github.com/p0p4k/pflowtts_pytorch): codes of MAS in training

[Bert-VITS2](https://github.com/Plachtaa/VITS-fast-fine-tuning) : numba version of MAS and modern pytorch codes of Vits

[fish-speech](https://github.com/fishaudio/fish-speech): dataclass usage and mel-spectrogram transforms using torchaudio, gradio webui

[gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS): melstyle encoder for voice clone

[coqui xtts](https://huggingface.co/spaces/coqui/xtts): gradio webui

Chinese Dirtionary Of DiffSinger: [Multi-langs_Dictionary](https://github.com/colstone/Multi-langs_Dictionary) and [atonyxu's fork](https://github.com/atonyxu/Multi-langs_Dictionary)

## TODO

- [x] Release pretrained models.
- [x] Support Japanese language.
- [x] User friendly preprocess and inference script.
- [x] Enhance documentation and citations.
- [x] Release multilingual checkpoint.

## Disclaimer

Any organization or individual is prohibited from using any technology in this repo to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.