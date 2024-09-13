import os
import json
from tqdm import tqdm
from dataclasses import dataclass, asdict

import torch
import torchaudio

from typing import List,Tuple

from subprocess import Popen
import platform
import psutil
import signal
import shutil
import sys
import requests
import torch,gc
from glob import glob

import gradio as gr

#get function for repo
from utils.audio import LogMelSpectrogram, load_and_resample_audio
from preprocess import g2p_mapping,load_filelist
from api import StableTTSAPI
from config import MelConfig

# webui 
import re
import numpy as np
import matplotlib.pyplot as plt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

system = platform.system()
python_executable = sys.executable or "python"

supported_languages = list(g2p_mapping.keys())

os.makedirs('./runs', exist_ok=True)
os.makedirs('./stableTTS_datasets', exist_ok=True)
os.makedirs("./checkpoints/pretrain", exist_ok=True)

model = None		
training_process = None    
tenserboard_process = None    

#settings
tts_model_path = './checkpoints/checkpoint_0.pt'
vocoder_model_path = './vocoders/pretrained/firefly-gan-base-generator.ckpt'
vocoder_type = 'ffgan'
current_model = ""

#models
pretrained_model_path = os.path.join("./checkpoints/pretrain", "checkpoint_0.pt")
url_model="https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/StableTTS/checkpoint_0.pt"

vocos_model_path = './vocoders/pretrained/vocos.pt'
url_vocos="https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/vocoders/vocos.pt"

firefly_model_path = './vocoders/pretrained/firefly-gan-base-generator.ckpt'
url_firefly='https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base-generator.ckpt'

#check model and models
def download_model_file(url: str, local_filename: str):
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        print(f"Downloading: {local_filename}")
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_filename, 'wb') as file, tqdm(
            total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
        print(f"File downloaded and saved as {local_filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

if not os.path.isfile(pretrained_model_path):
   download_model_file(url_model, pretrained_model_path)

if not os.path.isfile(vocos_model_path):
   download_model_file(url_vocos, vocos_model_path)

if not os.path.isfile(firefly_model_path):
   download_model_file(url_firefly, firefly_model_path)

#terminal
def terminate_process_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

def terminate_process(pid):
    if system == "Windows":
        cmd = f"taskkill /t /f /pid {pid}"
        os.system(cmd)
    else:
        terminate_process_tree(pid)

#preprocess 
def create_project(project_name: str):
    os.makedirs(f'./filelists/{project_name}', exist_ok=True)
    input_file_list = f'./filelists/{project_name}/filelist.txt'
    output_file_list = f'./filelists/{project_name}/filelist.json'
    output_feature_dir = f'./stableTTS_datasets/{project_name}'
    return [input_file_list, output_file_list, output_feature_dir]

def get_project_files(project_name: str):
    output_file_list = f'./filelists/{project_name}/filelist.json'
    run_dir = f'./runs/{project_name}'
    checkpoint_dir = f'./checkpoints/{project_name}'
    return [output_file_list, run_dir, checkpoint_dir]

def get_projects(folder_path=r'./filelists') -> List[str]:
    json_files = []
    for folder in os.listdir(folder_path):
        file_json = os.path.join(folder_path, folder, "filelist.json")
        if os.path.isfile(file_json):
            json_files.append(folder)
    return json_files

def refresh_projects() -> Tuple[List[str], str]:
    projects = get_projects()
    first_project = projects[0] if projects else None
    return projects, first_project

def preprocess_audio_files(input_file_list: str, output_feature_dir: str, output_file_list: str, language: str, should_resample:bool, progress=gr.Progress()):
    
    if not os.path.isfile(input_file_list):
        return f"No such file or directory: '{input_file_list}'"
    
    mel_config = MelConfig()

    mel_extractor = LogMelSpectrogram(**asdict(mel_config)).to(device)
    text_to_phoneme = g2p_mapping.get(language)
    
    output_mel_dir = os.path.join(output_feature_dir, 'mels')
    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file_list), exist_ok=True)
    
    if should_resample:
       output_wav_dir = os.path.join(output_feature_dir, 'waves')
       os.makedirs(output_wav_dir, exist_ok=True)
    
    @torch.inference_mode()
    def process_audio_file(line) -> str:
        idx, audio_path, text = line
        audio = load_and_resample_audio(audio_path, mel_config.sample_rate, device=device)
        if audio is not None: 
            audio_name, _ = os.path.splitext(os.path.basename(audio_path))
            
            try:
                phonemes = text_to_phoneme(text)
                if phonemes:
                    mel = mel_extractor(audio.to(device)).cpu().squeeze(0)
                    output_mel_path = os.path.join(output_mel_dir, f'{idx}_{audio_name}.pt')
                    torch.save(mel, output_mel_path)
                    
                    if should_resample:
                       audio_path = os.path.join(output_wav_dir, f'{idx}_{audio_name}.wav')
                       torchaudio.save(audio_path, audio.cpu(), mel_config.sample_rate)
                    return json.dumps({'mel_path': output_mel_path, 'phone': phonemes, 'audio_path': audio_path, 'text': text, 'mel_length': mel.size(-1)}, ensure_ascii=False, allow_nan=False)
            except Exception as e:
                print(f'Error processing {audio_path}: {str(e)}')
            
    input_file_list = load_filelist(input_file_list)
    processed_files = []

    for i, line in enumerate(progress.tqdm(input_file_list, desc="Processing files")):
        result = process_audio_file(line)
        if result:
           processed_files.append(f'{result}\n')
  
    with open(output_file_list, 'w', encoding='utf-8') as f:
        f.writelines(processed_files)
    
    return f"File list has been saved to {output_file_list}"

# train
def save_config(config, file_path):
    with open(file_path, 'w') as file:
        json.dump(config, file, indent=4)
    print(f"Configuration saved to {file_path}")

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def get_config_data(project_name):
    config_dir = os.path.join(r'./filelists',project_name)
    config_file_json = os.path.join(config_dir, "config.json")
    if os.path.isfile(config_file_json)==False:return 16,0.0001,200,16,1,200
    data=load_config(config_file_json) 

    return data["batch_size"],data["learning_rate"],data["num_epochs"],data["log_interval"],data["save_interval"],data["warmup_steps"]

def create_training_config(config_file,config_file_json,
        train_dataset_path: str = 'filelists/petta/filelist.json',
        test_dataset_path: str = 'filelists/ben/filelist.json', 
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10000,
        model_save_path: str = './checkpoints',
        log_dir: str = './runs',
        log_interval: int = 16,
        save_interval: int = 1,
        warmup_steps: int = 200,
    ):
    
    config_content = f"""
from dataclasses import dataclass

@dataclass
class MelConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    f_min: float = 0.0
    f_max: float = None
    pad: int = 0
    n_mels: int = 128
    center: bool = False
    pad_mode: str = "reflect"
    mel_scale: str = "slaney"
    
    def __post_init__(self):
        if self.pad == 0:
            self.pad = (self.n_fft - self.hop_length) // 2
            
@dataclass
class ModelConfig:
    hidden_channels: int = 256
    filter_channels: int = 1024
    n_heads: int = 4
    n_enc_layers: int = 3 
    n_dec_layers: int = 6 
    kernel_size: int = 3
    p_dropout: int = 0.1
    gin_channels: int = 256

@dataclass
class TrainConfig:
    train_dataset_path: str = "{train_dataset_path}"
    test_dataset_path: str = "{test_dataset_path}"
    batch_size: int = {batch_size}
    learning_rate: float = {learning_rate}
    num_epochs: int = {num_epochs}
    model_save_path: str = "{model_save_path}"
    log_dir: str = "{log_dir}"
    log_interval: int = {log_interval}
    save_interval: int = {save_interval}
    warmup_steps: int = {warmup_steps}
    
@dataclass
class VocosConfig:
    input_channels: int = 128
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8
"""
    with open(config_file, "w") as f:
        f.write(config_content)

    config = {
        "train_dataset_path": train_dataset_path,
        "test_dataset_path": test_dataset_path,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "model_save_path": model_save_path,
        "log_dir": log_dir,
        "log_interval": log_interval,
        "save_interval": save_interval,
        "warmup_steps": warmup_steps
    }

    save_config(config,config_file_json)

def train_model(train_dataset_path: str, batch_size:int, learning_rate:float, num_epochs:int, model_save_path:str, log_dir:str, log_interval:str, save_interval:int, warmup_steps:int, use_finetune:bool, finetune_model:str = r"./pretrain/checkpoint_0.pt"):
    config_dir = os.path.dirname(train_dataset_path)
    config_file = os.path.join(config_dir, "config.py")
    config_file_json = os.path.join(config_dir, "config.json")
 
    create_training_config(config_file,config_file_json, train_dataset_path, "", batch_size, learning_rate, num_epochs + 1, model_save_path, log_dir, log_interval, save_interval, warmup_steps)
    
    if os.path.isfile(config_file):
       shutil.copy(config_file, "config.py")
     
    if use_finetune:
       os.makedirs(model_save_path, exist_ok=True)
       finetune_model = os.path.join(model_save_path, "checkpoint_0.pt")
       if not os.path.isfile(finetune_model):
          shutil.copy(pretrained_model_path, finetune_model)

    yield "Training started !",gr.update(interactive=False),gr.update(interactive=True)

    clear_model()
    start_training()
    yield "Training finish !",gr.update(interactive=True),gr.update(interactive=False)

def clear_model():
    global model
    if model is not None:
       del model
       gc.collect()
       torch.cuda.empty_cache()    

def start_training():   
    global training_process
    if training_process is not None:return f"Train run already!"
    
    cmd = f"{python_executable} train.py"

    training_process = Popen(cmd, shell=True)
    training_process.wait()
    
def stop_training():
    global training_process
    if training_process is None:return f"Train not run !"
    terminate_process_tree(training_process.pid)
    training_process=None
    return "Training cancel !",gr.update(interactive=True),gr.update(interactive=False)

def refresh_dropdown_train():
    names,select=refresh_projects()
    return gr.Dropdown(choices=names,value=select, label="Project")

# tensorboard
def start_tensorboard(log_dir: str, port: int = 6006):
    global tenserboard_process 
    if tenserboard_process is not None:return f"Tensorboard run on port {port}",gr.update(interactive=False),gr.update(interactive=True),gr.update(interactive=True)

    try:
        cmd = f"tensorboard --logdir {log_dir} --port {port}"
        tenserboard_process = Popen(cmd, shell=True)
        yield f"TensorBoard started. Open http://localhost:{port} to view. {cmd}",gr.update(interactive=False),gr.update(interactive=True),gr.update(interactive=True)
        tenserboard_process.wait()
        
    except Exception as e:
        return f"Failed to start TensorBoard: {str(e)}",gr.update(interactive=False),gr.update(interactive=True),gr.update(interactive=False)

def stop_tensorboard():
    global tenserboard_process 
    if tenserboard_process is None:return f"Tensorboard not run !",gr.update(interactive=True),gr.update(interactive=False),gr.update(interactive=False)

    try:
        terminate_process_tree(tenserboard_process.pid)
        yield "Tensorboard stopped",gr.update(interactive=True),gr.update(interactive=False),gr.update(interactive=False)
        tenserboard_process=None 
    except Exception as e:
        return f"Failed to stop TensorBoard: {str(e)}",gr.update(interactive=True),gr.update(interactive=False),gr.update(interactive=False)
    
def get_tensorboard_projects(folder_path=r'./runs') -> List[str]:
    return os.listdir(folder_path)

def refresh_tensorboard_projects() -> Tuple[List[str], str]:
    projects = get_tensorboard_projects()
    first_project = projects[0] if projects else None
    return projects, first_project

def get_tensorboard_log_dir(project_name="", folder_path=r'./runs') -> str:
    return f"{folder_path}/{project_name}"

def refresh_dropdown_tensorboard(name):
    names,select=refresh_tensorboard_projects()
    return gr.Dropdown(choices=names,value=select, label="Project")

# interface
def update_model(tts_model_path:str, vocoder_model_path:str, vocoder_type:str):
    global model
    global current_model
    if current_model != tts_model_path:
       model = StableTTSAPI(tts_model_path, vocoder_model_path, vocoder_type).to(device)
       print("model change")

def get_checkpoints(folder: str) -> Tuple[List[str], str]:

    if not folder:
        return [], ""
    
    checkpoints_path = os.path.join('checkpoints', folder, "*.pt")
    checkpoints = glob(checkpoints_path) 
    checkpoint_names = [os.path.basename(item) for item in checkpoints if 'checkpoint' in os.path.basename(item)]   
    checkpoint_names.sort()
    
    if checkpoint_names:
        return checkpoint_names, checkpoint_names[0]
    
    return [], ""

def refresh_dropdown_checkpoints(folder):
    names,select=get_checkpoints(folder)
    return gr.Dropdown(choices=names,value=select, label="Checkpoint")

def refresh_dropdown():
    names,select=refresh_projects()
    return gr.Dropdown(choices=names,value=select, label="Project")

@torch.inference_mode()
def generate_speech(text, ref_audio, language, step, temperature, length_scale, solver, cfg):
    text = remove_newlines_after_punctuation(text)
    
    if language == 'chinese':
        text = text.replace(' ', '')
        
    audio, mel = model.inference(text, ref_audio, language, step, temperature, length_scale, solver, cfg)
    
    max_val = torch.max(torch.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    
    audio_output = (model.mel_config.sample_rate, (audio.cpu().squeeze(0).numpy() * 32767).astype(np.int16))
    mel_output = plot_mel_spectrogram(mel.cpu().squeeze(0).numpy())
    
    return audio_output, mel_output

def plot_mel_spectrogram(mel_spectrogram):
    plt.close()  # prevent memory leak
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove white edges
    return fig

def remove_newlines_after_punctuation(text):
    pattern = r'([，。！？、""''《》【】；：,.!?\'\"<>()\[\]{}])\n'
    return re.sub(pattern, r'\1', text)

def set_seed(seed):
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed

@torch.inference_mode()
def generate_tts(folder, checkpoint, text, ref_audio, language, step, temperature, length_scale, solver, cfg,seed=-1,random=True):
    if random:seed=-1
    seed = set_seed(seed)
  
    update_model(os.path.join("checkpoints", folder, checkpoint), vocoder_model_path, vocoder_type)

    text = remove_newlines_after_punctuation(text)
    
    if language == 'chinese':
        text = text.replace(' ', '')
        
    audio, mel = model.inference(text, ref_audio, language, step, temperature, length_scale, solver, cfg)
    
    max_val = torch.max(torch.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    
    audio_output = (model.mel_config.sample_rate, (audio.cpu().squeeze(0).numpy() * 32767).astype(np.int16))
    mel_output = plot_mel_spectrogram(mel.cpu().squeeze(0).numpy())
    
    return audio_output, mel_output , seed

def check_and_download(pretrained_model_path, vocos_model_path, firefly_model_path):
    
    if not os.path.isfile(pretrained_model_path):
        footer_message = "Downloading Pretrained Model... "
        print(footer_message)
        yield gr.update(value=f"<footer style='text-align: center; padding: 10px; font-size: 14px; color: grey; border-top: 1px solid #eaeaea;'>{footer_message}</footer>")
        download_model_file(url_model, pretrained_model_path)
    
    if not os.path.isfile(vocos_model_path):
        footer_message = "Downloading Vocos Model... "
        print(footer_message)
        yield gr.update(value=f"<footer style='text-align: center; padding: 10px; font-size: 14px; color: grey; border-top: 1px solid #eaeaea;'>{footer_message}</footer>")
        download_model_file(url_vocos, vocos_model_path)

    if not os.path.isfile(firefly_model_path):
        footer_message = "Downloading Firefly Model... "
        print(footer_message)
        yield gr.update(value=f"<footer style='text-align: center; padding: 10px; font-size: 14px; color: grey; border-top: 1px solid #eaeaea;'>{footer_message}</footer>")
        download_model_file(url_firefly, firefly_model_path)

    footer_message += "All models downloaded!"
    yield gr.update(value=f"<footer style='text-align: center; padding: 10px; font-size: 14px; color: grey; border-top: 1px solid #eaeaea;'>{footer_message}</footer>")

def get_file():
    file_path = gr.File.file_select()
    return file_path

def launch_tensorboard(port):
    import webbrowser
    url = f"http://localhost:{port}"
    webbrowser.open(url, new=2)
    return f"TensorBoard launched at {url}"

def button_enable():
    return gr.update(interactive=True)

def button_disable():
    return gr.update(interactive=False)

def random_sample(nameproject):
    filelist_file = os.path.join('filelists',nameproject, "filelist.json")

    if os.path.isfile(filelist_file)==False:return "",None
    
    with open(filelist_file, 'r') as file:
         data = [json.loads(line) for line in file]

    entry = random.choice(data)

    return entry["text"],entry["audio_path"]

def enable_button_setting(value):
    return gr.update(interactive=value)

def create_interface():

    with gr.Blocks() as app:
        gui_title = 'StableTTS ALL IN ONE'
        gui_description = """Next-generation TTS model using flow-matching and DiT, inspired by Stable Diffusion 3."""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"# {gui_title}")
                gr.Markdown(gui_description)
     
        with gr.Tabs():

            with gr.TabItem("Preprocess Data"):
                with gr.Row():
                    project_name = gr.Textbox(label="Project Name", value='test')
                    create_project_btn = gr.Button("Create")

                with gr.Row():
                    input_file_list = gr.Textbox(label="Input File List Path", value='./filelists/filelist.txt',interactive=False)
                    output_file_list = gr.Textbox(label="Output File List Path", value='./filelists/filelist.json',interactive=False)
                    output_feature_dir = gr.Textbox(label="Output Feature Directory", value='./stableTTS_datasets',interactive=False)
                    
                language = gr.Dropdown(label="Language", choices=supported_languages, value="english")

                resample_audio = gr.Checkbox(label="Resample Audio", value=False)
                preprocess_output = gr.Textbox(label="Preprocess Output", lines=4)
                
                preprocess_btn = gr.Button("Preprocess Data",interactive=False)

                preprocess_btn.click(
                    fn=preprocess_audio_files,
                    inputs=[input_file_list, output_feature_dir, output_file_list, language, resample_audio],
                    outputs=preprocess_output
                )
           
            create_project_btn.click(fn=create_project, inputs=[project_name], outputs=[input_file_list, output_file_list, output_feature_dir])
            create_project_btn.click(fn=button_enable, outputs=[preprocess_btn])

     
            with gr.TabItem("Train Model"):
                initial_projects = get_projects()
                initial_project = initial_projects[0] if initial_projects else None

                if initial_project is not None:
                    train_value, log_value, model_value = get_project_files(initial_project)
                else:
                    train_value, log_value, model_value = './filelists/filelist.json', './runs', './checkpoints' 

                with gr.Row():
                    project_dropdown = gr.Dropdown(choices=initial_projects, value=initial_project, label="Project", interactive=True,allow_custom_value=True)
                    refresh_projects_btn = gr.Button("Refresh Projects")             
                    
                with gr.Row():
                    train_dataset_path = gr.Textbox(label="Train Dataset Path", value=train_value,interactive=False)
                    log_dir = gr.Textbox(label="Log Directory", value=log_value,interactive=False)
                    model_save_path = gr.Textbox(label="Model Save Path", value=model_value,interactive=False)
                
                with gr.Row():
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=128, step=1, value=16)
                    num_epochs = gr.Slider(label="Number of Epochs", minimum=1, maximum=10000, step=1, value=200)
                    warmup_steps = gr.Slider(label="Warmup Steps", minimum=1, maximum=10000, step=1, value=200)
                
                with gr.Row():
                    log_interval = gr.Slider(label="Log Interval", minimum=1, maximum=100, step=1, value=16)
                    save_interval = gr.Slider(label="Save Interval", minimum=1, maximum=100, step=1, value=1)
                
                learning_rate = gr.Number(label="Learning Rate", value=1e-4)
                
                pretrain_model_path = gr.Textbox(label="Pretrain Model Path", value=pretrained_model_path)
        

                use_finetune = gr.Checkbox(label="Use Finetune", value=True)     

                train_output = gr.Textbox(label="Training Output", lines=4)
                
                with gr.Row():

                    train_start_model_btn = gr.Button("Start Train",interactive=False)
                    train_stop_model_btn = gr.Button("Stop Train",interactive=False)               
   
            refresh_projects_btn.click(fn=get_config_data,inputs=project_dropdown,outputs=[ batch_size, learning_rate,num_epochs,log_interval, save_interval, warmup_steps])
            project_dropdown.change(fn=get_config_data, inputs=project_dropdown, outputs=[ batch_size, learning_rate,num_epochs,log_interval, save_interval, warmup_steps])

            refresh_projects_btn.click(fn=refresh_projects, outputs=[project_dropdown, project_dropdown])
            project_dropdown.change(fn=get_project_files, inputs=project_dropdown, outputs=[train_dataset_path, log_dir, model_save_path])
            
            refresh_projects_btn.click(fn=button_enable, outputs=[train_start_model_btn])
            project_dropdown.change(fn=button_enable, outputs=[train_start_model_btn])
             
            train_start_model_btn.click(
                  fn=train_model,
                  inputs=[train_dataset_path, batch_size, learning_rate,num_epochs, model_save_path, log_dir, log_interval, save_interval, warmup_steps, use_finetune, pretrain_model_path],
                  outputs=[train_output,train_start_model_btn,train_stop_model_btn])
           
            train_stop_model_btn.click(fn=stop_training,outputs=[train_output,train_start_model_btn,train_stop_model_btn])
            
            refresh_projects_btn.click(fn=refresh_dropdown_train, outputs=[project_dropdown]) 

            with gr.TabItem("Tensorboard"):
                with gr.Row():
                    initial_tensorboard_projects = get_tensorboard_projects()
                    initial_tensorboard_project = initial_tensorboard_projects[0] if initial_tensorboard_projects else None
                    tensorboard_project_dropdown = gr.Dropdown(choices=initial_tensorboard_projects, value=initial_tensorboard_project, label="Project", interactive=True,allow_custom_value=True)
                    refresh_tensorboard_btn = gr.Button("Refresh Tensorboard")             
                    refresh_tensorboard_btn.click(fn=refresh_tensorboard_projects, outputs=[tensorboard_project_dropdown, tensorboard_project_dropdown])

                    if initial_tensorboard_project is not None:
                        log_value = get_tensorboard_log_dir(initial_tensorboard_project)
                    else:
                        log_value = './runs'

                tensorboard_log_path = gr.Textbox(label="Tensorboard Log Directory", value=log_value,interactive=False)

                tensorboard_project_dropdown.change(fn=get_tensorboard_log_dir, inputs=tensorboard_project_dropdown, outputs=[tensorboard_log_path])
               
                with gr.Row():    
                     port_tensorboard = gr.Number(label="Port",value=6006)
                     start_tensorboard_btn = gr.Button("Start TensorBoard",interactive=False)
                     stop_tensorboard_btn = gr.Button("Stop TensorBoard",interactive=False)
                     open_tensorboard_btn = gr.Button("Open TensorBoard",interactive=False)

    
                tensorboard_output = gr.Textbox(label="TensorBoard Output", lines=2)

                start_tensorboard_btn.click(
                    fn=start_tensorboard,
                    inputs=[tensorboard_log_path,port_tensorboard],
                    outputs=[tensorboard_output,start_tensorboard_btn,stop_tensorboard_btn,open_tensorboard_btn],
                )
                
                stop_tensorboard_btn.click(
                    fn=stop_tensorboard,
                    outputs=[tensorboard_output,start_tensorboard_btn,stop_tensorboard_btn,open_tensorboard_btn]
                )

                open_tensorboard_btn.click(
                    fn=launch_tensorboard,
                    inputs=[port_tensorboard],
                )

                refresh_tensorboard_btn.click(button_enable,outputs=start_tensorboard_btn)
                refresh_tensorboard_btn.click(button_disable,outputs=stop_tensorboard_btn)
                refresh_tensorboard_btn.click(button_disable,outputs=open_tensorboard_btn)

                tensorboard_project_dropdown.change(fn=refresh_dropdown_tensorboard , outputs=[tensorboard_project_dropdown]) 


            with gr.TabItem("Interface"):
                with gr.Blocks(theme=gr.themes.Base()) as demo:
                    demo.load(None, None, js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}")
            
                    with gr.Row():
                        with gr.Column():
                            initial_checkpoints, initial_checkpoint = get_checkpoints(initial_project)
    
                            with gr.Row():
                                model_project_dropdown = gr.Dropdown(choices=initial_projects, value=initial_project, label="Project", interactive=True,allow_custom_value=True)
                                model_checkpoint_dropdown = gr.Dropdown(choices=initial_checkpoints, value=initial_checkpoint, label="Checkpoint", interactive=True,allow_custom_value=True)

                            refresh_model_btn = gr.Button("Refresh Projects")             
                            
                            random_model_btn = gr.Button("Random Sample")
                            
                            input_text = gr.Textbox(
                                label="Input Text",
                                info="Enter your text here",
                            )
                         
                            reference_audio = gr.Audio(
                                label="Reference Audio",
                                type="filepath"
                            )

              
                            language_dropdown = gr.Dropdown(
                                label='Language',
                                choices=supported_languages,
                                value='english'
                            )
                            
                            step_slider = gr.Slider(
                                label='Step',
                                minimum=1,
                                maximum=100,
                                value=25,
                                step=1
                            )
                            
                            temperature_slider = gr.Slider(
                                label='Temperature',
                                minimum=0,
                                maximum=2,
                                value=1,
                            )
                            
                            length_scale_slider = gr.Slider(
                                label='Length Scale',
                                minimum=0,
                                maximum=5,
                                value=1,
                            )
                            
                            solver_dropdown = gr.Dropdown(
                                label='ODE Solver',
                                choices=['euler', 'midpoint', 'dopri5', 'rk4', 'implicit_adams', 'bosh3', 'fehlberg2', 'adaptive_heun'],
                                value='dopri5'
                            )
                            
                            cfg_slider = gr.Slider(
                                label='CFG',
                                minimum=0,
                                maximum=10,
                                value=3,
                            )
                            
                            with gr.Row():
                                 
                                 seed_bool = gr.Checkbox(
                                     label='Random',
                                     value=True
                                 )
     
                                 seed_value = gr.Slider(
                                     label='Seeds',
                                     step=1,
                                     minimum=0,
                                     maximum=100000000,
                                     value=0,
                                     interactive=True
                                 )

                            
                        with gr.Column():
                            mel_plot = gr.Plot(label="Mel Spectrogram Visualization")
                            generated_audio = gr.Audio(label="Generated Audio", autoplay=True)
                            generate_btn = gr.Button("Generate", elem_id="send-btn", visible=True, variant="primary")
            
                        refresh_model_btn.click(fn=refresh_projects, outputs=[model_project_dropdown, model_project_dropdown])  
                        refresh_model_btn.click(fn=get_checkpoints, inputs=[model_project_dropdown], outputs=[model_checkpoint_dropdown, model_checkpoint_dropdown])  
                        refresh_model_btn.click(fn=refresh_dropdown_checkpoints, inputs=[model_project_dropdown], outputs=[model_checkpoint_dropdown])  
                        refresh_model_btn.click(fn=refresh_dropdown,  outputs=[model_project_dropdown])  

                        model_project_dropdown.change(fn=refresh_dropdown_checkpoints, inputs=[model_project_dropdown], outputs=[model_checkpoint_dropdown]) 
                         
                        random_model_btn.click(fn=random_sample,inputs=[model_project_dropdown],outputs=[input_text,reference_audio])
 

                    generate_btn.click(generate_tts, [model_project_dropdown, model_checkpoint_dropdown, input_text, reference_audio, language_dropdown, step_slider, temperature_slider, length_scale_slider, solver_dropdown, cfg_slider,seed_value,seed_bool], outputs=[generated_audio, mel_plot,seed_value])
                    
                    seed_bool.change(fn=enable_button_setting,inputs=seed_bool,outputs=seed_value)
      


        footer = gr.HTML(f"""
         <footer style="text-align: center; padding: 10px; font-size: 14px; color: grey; border-top: 1px solid #eaeaea;">
             All set! The models are ready to go. Running on <strong>{device}</strong>.
         </footer>
         """)

    return app

if __name__ == "__main__":
    create_interface().launch(debug=True)