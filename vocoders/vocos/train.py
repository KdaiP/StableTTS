import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import itertools
from dataclasses import asdict

from models.model import Vocos
from dataset import VocosDataset
from models.discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from models.loss import feature_loss, generator_loss, discriminator_loss, MultiScaleMelSpectrogramLoss, SingleScaleMelSpectrogramLoss
from config import MelConfig, VocosConfig, TrainConfig
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.load import continue_training

torch.backends.cudnn.benchmark = True
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def _init_config(vocos_config: VocosConfig, mel_config: MelConfig, train_config: TrainConfig):
    if vocos_config.input_channels != mel_config.n_mels:
        raise ValueError("input_channels and n_mels must be equal.")
    
    if not os.path.exists(train_config.model_save_path):
        print(f'Creating {train_config.model_save_path}')
        os.makedirs(train_config.model_save_path, exist_ok=True)

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    vocos_config = VocosConfig()
    mel_config = MelConfig()
    train_config = TrainConfig()
    
    _init_config(vocos_config, mel_config, train_config)
    
    generator = Vocos(vocos_config, mel_config).to(rank)
    mpd = MultiPeriodDiscriminator().to(rank)
    mrd = MultiResolutionDiscriminator().to(rank)
    loss_fn = MultiScaleMelSpectrogramLoss().to(rank)
    if rank == 0:
        print(f"Generator params: {sum(p.numel() for p in generator.parameters()) / 1e6}")
        print(f"Discriminator mpd params: {sum(p.numel() for p in mpd.parameters()) / 1e6}")
        print(f"Discriminator mrd params: {sum(p.numel() for p in mrd.parameters()) / 1e6}")
    
    generator = DDP(generator, device_ids=[rank])
    mpd = DDP(mpd, device_ids=[rank])
    mrd = DDP(mrd, device_ids=[rank])

    train_dataset = VocosDataset(train_config.train_dataset_path, train_config.segment_size, mel_config)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_config.batch_size, num_workers=4, pin_memory=False, persistent_workers=True)
    
    if rank == 0:
        writer = SummaryWriter(train_config.log_dir)

    optimizer_g = optim.AdamW(generator.parameters(), lr=train_config.learning_rate)
    optimizer_d = optim.AdamW(itertools.chain(mpd.parameters(), mrd.parameters()), lr=train_config.learning_rate)
    scheduler_g = get_cosine_schedule_with_warmup(optimizer_g, num_warmup_steps=int(train_config.warmup_steps), num_training_steps=train_config.num_epochs * len(train_dataloader))
    scheduler_d = get_cosine_schedule_with_warmup(optimizer_d, num_warmup_steps=int(train_config.warmup_steps), num_training_steps=train_config.num_epochs * len(train_dataloader))
    
    # load latest checkpoints if possible
    current_epoch = continue_training(train_config.model_save_path, generator, mpd, mrd, optimizer_d, optimizer_g)

    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(current_epoch, train_config.num_epochs):  # loop over the train_dataset multiple times
        train_dataloader.sampler.set_epoch(epoch)
        if rank == 0:
            dataloader = tqdm(train_dataloader)
        else:
            dataloader = train_dataloader
            
        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(rank, non_blocking=True) for data in datas]
            audios, mels = datas
            audios_fake = generator(mels).unsqueeze(1) # shape: [batch_size, 1, segment_size]
            optimizer_d.zero_grad()
            
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(audios,audios_fake.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            
            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(audios,audios_fake.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            
            grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1000)
            grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1000)
            optimizer_d.step()
            scheduler_d.step()
            
            # generator
            optimizer_g.zero_grad()
            loss_mel = loss_fn(audios, audios_fake) * train_config.mel_loss_factor
            
            # MPD loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(audios,audios_fake)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            # MRD loss
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(audios,audios_fake)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            loss_gen_all.backward()
            
            grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1000)
            optimizer_g.step()
            scheduler_g.step()
            
            if rank == 0 and batch_idx % train_config.log_interval == 0:
                steps = epoch * len(dataloader) + batch_idx
                writer.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                writer.add_scalar("training/fm_loss_mpd", loss_fm_f.item(), steps)
                writer.add_scalar("training/gen_loss_mpd", loss_gen_f.item(), steps)
                writer.add_scalar("training/disc_loss_mpd", loss_disc_f.item(), steps)
                writer.add_scalar("training/fm_loss_mrd", loss_fm_s.item(), steps)
                writer.add_scalar("training/gen_loss_mrd", loss_gen_s.item(), steps)
                writer.add_scalar("training/disc_loss_mrd", loss_disc_s.item(), steps)
                writer.add_scalar("training/mel_loss", loss_mel.item(), steps)
                writer.add_scalar("grad_norm/grad_norm_mpd", grad_norm_mpd, steps)
                writer.add_scalar("grad_norm/grad_norm_mrd", grad_norm_mrd, steps)
                writer.add_scalar("grad_norm/grad_norm_g", grad_norm_g, steps)
                writer.add_scalar("learning_rate/learning_rate_d", scheduler_d.get_last_lr()[0], steps)
                writer.add_scalar("learning_rate/learning_rate_g", scheduler_g.get_last_lr()[0], steps)
            
        if rank == 0:
            torch.save(generator.module.state_dict(), os.path.join(train_config.model_save_path, f'generator_{epoch}.pt'))
            torch.save(mpd.module.state_dict(), os.path.join(train_config.model_save_path, f'mpd_{epoch}.pt'))
            torch.save(mrd.module.state_dict(), os.path.join(train_config.model_save_path, f'mrd_{epoch}.pt'))
            torch.save(optimizer_d.state_dict(), os.path.join(train_config.model_save_path, f'optimizerd_{epoch}.pt'))
            torch.save(optimizer_g.state_dict(), os.path.join(train_config.model_save_path, f'optimizerg_{epoch}.pt'))
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss_gen_all.item()}")

    cleanup()
    
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)