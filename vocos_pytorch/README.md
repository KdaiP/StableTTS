<div align="center">

# Vocos for StableTTS

Modified from the official implementation of [Vocos](https://github.com/gemelo-ai/vocos/tree/main). 

</div>

## Introduction

Vocos is a fast neural vocoder designed to synthesize audio waveforms from acoustic features. Trained using a Generative Adversarial Network (GAN) objective, Vocos can generate waveforms in a single forward pass. Unlike other typical GAN-based vocoders, Vocos does not model audio samples in the time domain. Instead, it generates spectral coefficients, facilitating rapid audio reconstruction through inverse Fourier transform.


## Inference

For detailed inference instructions, please refer to `inference.ipynb`

## Training

Setting up and training your model with Vocos is straightforward. Follow these steps to get started:

### Preparing Your Data

1. **Configure Data Settings**: Update the `DataConfig` in `preprocess.py`. Specifically, adjust the audio_dir to point to your collection of audio files.

2. **Run Preprocessing**: Run `preprocess.py`. This script will search (glob) for all audio files in the specified directory, resample them to the target sample_rate (modifiable in config.py), and generate a file list for training.

### Start training

1. **Adjust Training Configuration**: Edit `TrainConfig` in `config.py` to specify the file list path and tweak training hyperparameters to your needs.

2. **Start the Training Process**: Launch `train.py` to begin training your model.

### Experiment with Configurations

Feel free to explore and modify settings in `config.py` to modify the hyperparameters of vocos!


## References

[Vocos](https://github.com/gemelo-ai/vocos/tree/main)