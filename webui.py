from dataclasses import asdict
from text import symbols
import torch
import torchaudio
from IPython.display import Audio, display

from utils.audio import LogMelSpectrogram
from config import ModelConfig, VocosConfig, MelConfig
from models.model import StableTTS
from vocos_pytorch.models.model import Vocos
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text import cleaned_text_to_sequence
from datas.dataset import intersperse

import gradio as gr

@ torch.inference_mode()
def inference(text: str, ref_audio: torch.Tensor, tts_model: StableTTS, mel_extractor: LogMelSpectrogram, vocoder: Vocos, phonemizer, sample_rate: int, step: int=10) -> torch.Tensor:
    x = torch.tensor(intersperse(cleaned_text_to_sequence(phonemizer(text)), item=0), dtype=torch.long).unsqueeze(0)
    x_len = torch.tensor([x.size(-1)], dtype=torch.long)
    waveform, sr = torchaudio.load(ref_audio)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    y = mel_extractor(waveform)
    print(y.shape)
    mel = tts_model.synthesise(x, x_len, step, y=y, length_scale=1)['decoder_outputs']
    audio = vocoder(mel)
    return audio, mel

def get_pipeline(n_vocab: int, tts_model_config: ModelConfig, mel_config: MelConfig, vocoder_config: VocosConfig, tts_checkpoint_path, vocoder_checkpoint_path):
    tts_model = StableTTS(n_vocab, mel_config.n_mels, **asdict(tts_model_config))
    mel_extractor = LogMelSpectrogram(mel_config)
    vocoder = Vocos(vocoder_config, mel_config)
    tts_model.load_state_dict(torch.load(tts_checkpoint_path, map_location='cpu'))
    vocoder.load_state_dict(torch.load(vocoder_checkpoint_path, map_location='cpu'))
    return tts_model, mel_extractor, vocoder

def stable_tts(text, reference_audio, tts_checkpoint_path='./pretrained_checkpoints/stabletts_pretrained.pt', vocoder_checkpoint_path='./pretrained_checkpoints/vocos_pretrained.pt'):
    tts_model_config = ModelConfig()
    mel_config = MelConfig()
    vocoder_config = VocosConfig()

    tts_model, mel_extractor, vocoder = get_pipeline(len(symbols), 
                                                     tts_model_config=tts_model_config, 
                                                     mel_config=mel_config, 
                                                     vocoder_config=vocoder_config, 
                                                     tts_checkpoint_path=tts_checkpoint_path, 
                                                     vocoder_checkpoint_path=vocoder_checkpoint_path)
    
    total_params = sum(p.numel() for p in tts_model.parameters()) / 1e6
    print(total_params)
    chinese = True
    phonemizer = chinese_to_cnm3 if chinese else english_to_ipa2

    output, mel = inference(text=text, 
                            ref_audio=reference_audio, 
                            tts_model=tts_model, 
                            mel_extractor=mel_extractor, 
                            vocoder=vocoder, 
                            phonemizer=phonemizer, 
                            sample_rate=mel_config.sample_rate, 
                            step=15)
    return output  


app = gr.Interface(
    fn=stable_tts,
    inputs=[
        gr.Textbox(lines=4, label="Input Text"),
        gr.File(label="Reference audio"),
        gr.File(label="TTS checkpoint path"),
        gr.File(label="Vocoder checkpoint path")
    ],
    outputs=gr.Audio(label="Generated Voice")
)

app.launch()

