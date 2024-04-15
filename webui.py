import os
os.environ['TMPDIR'] = './temps' # avoid the system default temp folder not having access permissions
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # use huggingfacae mirror for users that could not login to huggingface

from dataclasses import asdict
from text import symbols
import torch
import torchaudio

from utils.audio import LogMelSpectrogram
from config import ModelConfig, VocosConfig, MelConfig
from models.model import StableTTS
from vocos_pytorch.models.model import Vocos
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text import cleaned_text_to_sequence
from datas.dataset import intersperse

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@ torch.inference_mode()
def inference(text: str, ref_audio: torch.Tensor, language: str, checkpoint_path: str, step: int=10) -> torch.Tensor:
    global last_checkpoint_path
    if checkpoint_path != last_checkpoint_path:
        tts_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')) 
        last_checkpoint_path = checkpoint_path
        
    phonemizer = chinese_to_cnm3 if language == 'chinese' else english_to_ipa2
    
    # prepare input for tts model
    x = torch.tensor(intersperse(cleaned_text_to_sequence(phonemizer(text)), item=0), dtype=torch.long, device=device).unsqueeze(0)
    x_len = torch.tensor([x.size(-1)], dtype=torch.long, device=device)
    waveform, sr = torchaudio.load(ref_audio)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    y = mel_extractor(waveform).to(device)
    
    # inference
    mel = tts_model.synthesise(x, x_len, step, y=y, temperature=1, length_scale=1)['decoder_outputs']
    audio = vocoder(mel)
    
    # process output for gradio
    audio_output = (sample_rate, (audio.cpu().squeeze(0).numpy() * 32767).astype(np.int16)) # (samplerate, int16 audio) for gr.Audio
    mel_output = plot_mel_spectrogram(mel.cpu().squeeze(0).numpy()) # get the plot of mel
    return audio_output, mel_output

def get_pipeline(n_vocab: int, tts_model_config: ModelConfig, mel_config: MelConfig, vocoder_config: VocosConfig, tts_checkpoint_path, vocoder_checkpoint_path):
    tts_model = StableTTS(n_vocab, mel_config.n_mels, **asdict(tts_model_config))
    mel_extractor = LogMelSpectrogram(mel_config)
    vocoder = Vocos(vocoder_config, mel_config)
    # tts_model.load_state_dict(torch.load(tts_checkpoint_path, map_location='cpu'))
    tts_model.to(device)
    tts_model.eval()
    vocoder.load_state_dict(torch.load(vocoder_checkpoint_path, map_location='cpu'))
    vocoder.to(device)
    vocoder.eval()
    return tts_model, mel_extractor, vocoder

def plot_mel_spectrogram(mel_spectrogram):
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # remove white edges
    return fig


def main():
    tts_model_config = ModelConfig()
    mel_config = MelConfig()
    vocoder_config = VocosConfig()

    tts_checkpoint_path = './checkpoints' # the folder that contains StableTTS checkpoints
    vocoder_checkpoint_path = './checkpoints/vocoder.pt'

    global tts_model, mel_extractor, vocoder, sample_rate, last_checkpoint_path
    sample_rate = mel_config.sample_rate
    last_checkpoint_path = None
    tts_model, mel_extractor, vocoder = get_pipeline(len(symbols), tts_model_config, mel_config, vocoder_config, tts_checkpoint_path, vocoder_checkpoint_path)
    
    tts_checkpoint_path = [path for path in Path(tts_checkpoint_path).rglob('*.pt') if 'optimizer' and 'vocoder' not in path.name]

    # gradio wabui
    gui_title = 'StableTTS'
    gui_description = """Next-generation TTS model using flow-matching and DiT, inspired by Stable Diffusion 3."""
    with gr.Blocks(analytics_enabled=False) as demo:

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"# {gui_title}")
                gr.Markdown(gui_description)

        with gr.Row():
            with gr.Column():
                input_text_gr = gr.Textbox(
                    label="Input Text",
                    info="One or two sentences at a time is better. Up to 200 text characters.",
                    value="你好，世界！",
                )
             
                ref_audio_gr = gr.Audio(
                    label="Reference Speaker",
                    type="filepath"
                )
                
                language_gr = gr.Dropdown(
                    label='Language',
                    choices=['chinese', 'english'],
                    value = 'chinese'
                )
                
                checkpoint_gr = gr.Dropdown(
                    label='checkpoint',
                    choices=tts_checkpoint_path,
                    value = tts_checkpoint_path[0]
                )
                
                step_gr = gr.Slider(
                    label='Step',
                    minimum=1,
                    maximum=100,
                    value=25,
                    step=1
                )


                tts_button = gr.Button("Send", elem_id="send-btn", visible=True)
                
            with gr.Column():
                mel_gr = gr.Plot(label="Mel Visual")
                audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)

        tts_button.click(inference, [input_text_gr, ref_audio_gr, language_gr, checkpoint_gr, step_gr], outputs=[audio_gr, mel_gr])

    demo.queue()  
    demo.launch(debug=True, show_api=True)


if __name__ == '__main__':
    main()