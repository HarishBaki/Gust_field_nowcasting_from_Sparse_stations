# %%
# === Top level imports ===
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR
from torch.amp import autocast, GradScaler


import xarray as xr
import zarr
import numpy as np
import math
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time

import os, shutil
import wandb, argparse, sys
from tqdm import tqdm
from pathlib import Path

from data_loader import nowcast_dataset, Transform
from models.Google_Unet import GoogleUNet
from models.Deep_CNN import DCNN
from models.UNet import UNet
from models.SwinT2_UNet import SwinT2UNet

from models.util import initialize_weights_xavier,initialize_weights_he
from models import predrnn_v2
from nowcastnet.layers.utils import warp, make_grid
from nowcastnet.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from nowcastnet.layers.evolution.evolution_network import Evolution_Network
from nowcastnet.layers.generation.noise_projector import Noise_Projector
from nowcastnet.models.nowcastnet import Net

from losses import MaskedErrorLoss, MaskedTVLoss, MaskedCharbonnierLoss, MaskedCombinedMAEQuantileLoss

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store, reshape_patch, reshape_patch_back
from types import SimpleNamespace

# %%
def train_evolution_net(train_dataloader, train_sampler,args):
    # === Model ===
    # %%
    model = Evolution_Network(args.input_sequence_length, args.output_sequence_length, base_c=32).to(args.device)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    # === Optimizer, scheduler, and early stopping ===
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < 200000:
            return 1.0  # keep lr = 1e-3 for first 200k steps
        else:
            return 0.1  # after 200k steps, reduce lr to 1e-4
    scheduler = LambdaLR(optimizer, lr_lambda)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Model created and moved to device.")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_step = 0
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "evo_net_latest.pt")
    if args.resume and os.path.exists(latest_ckpt_path):
        model, optimizer, scheduler, start_step = restore_model_checkpoint(model, optimizer, scheduler, latest_ckpt_path, args.device)
    
    scaler = GradScaler("cuda")   # <<< AMP scaler
    sample_tensor = torch.zeros(1, 1, args.img_size[0], args.img_size[1], device=args.device)
    grid = make_grid(sample_tensor)
    kx = torch.tensor([[1., 0., -1.],
                    [2., 0., -2.],
                    [1., 0., -1.]], device=args.device).view(1,1,3,3)
    ky = torch.tensor([[1.,  2.,  1.],
                    [0.,  0.,  0.],
                    [-1., -2., -1.]], device=args.device).view(1,1,3,3)
    # %%
    # === Iteration setup ===
    max_iters = 300_000  # total number of iterations to train
    global_step = start_step

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training for {max_iters} iterations (ignoring num_epochs).")
    
    while global_step < max_iters:
        if train_sampler is not None:
            # reshuffle at the start of each new "epoch"
            train_sampler.set_epoch(global_step // len(train_dataloader))
        # === Training ===
        model.train()
        show_progress = not dist.is_initialized() or dist.get_rank() == 0
        train_bar = tqdm(train_dataloader,desc=f"[Step {global_step}/{max_iters}] [Train]",leave=False) if show_progress else train_dataloader
        for batch_idx,batch in enumerate(train_bar):
            # %%
            global_step += 1
            if global_step > max_iters:
                break
            input_tensor, target_tensor,_,_ = batch
            input_tensor = input_tensor.to(args.device)  # [B, T_in, H, W]
            target_tensor = target_tensor.to(args.device)  # [B, T_out, H, W]

            B = input_tensor.shape[0]

            # === AMP forward/backward ===
            with autocast("cuda", dtype=torch.float16):   # <<< AMP autocast
                intensity, motion = model(input_tensor) # [B, output_seq_len, H, W], [B, output_seq_len*2, H, W]
                motion_ = motion.reshape(B, args.output_sequence_length, 2, args.img_size[0], args.img_size[1]) # [B, output_seq_len, 2, H, W]
                intensity_ = intensity.reshape(B, args.output_sequence_length, 1, args.img_size[0], args.img_size[1])   # [B, output_seq_len, 1, H, W]
                prime_bil_series = []   # gathers the x'(t)_bil
                double_prime_series = []    # gathers the x''(t)
                x0 = input_tensor[:, -1:]  # [B, 1, H, W], x0 according to the paper notation
                grid_extend = grid.repeat(B, 1, 1, 1)   # [B, 2, H, W]
                for i in range(args.output_sequence_length):
                    x_prime_bil = warp(x0, motion_[:, i], grid_extend, mode="bilinear", padding_mode="border")   # [B, 1, H, W]
                    prime_bil_series.append(x_prime_bil)
                    x_prime = warp(x0, motion_[:, i], grid_extend, mode="nearest", padding_mode="border")   # [B, 1, H, W]
                    x_double_prime = x_prime + intensity_[:, i]   # [B, 1, H, W]   
                    double_prime_series.append(x_double_prime)
                    # === STOP GRADIENT before feeding into next step ===
                    x0 = x_double_prime.detach()
                # %%
                x_prime_bil = torch.cat(prime_bil_series, dim=1)
                x_double_pr = torch.cat(double_prime_series, dim=1)

                # %%
                # ---- Accumulated L1 Loss ----
                loss_accum = F.l1_loss(x_prime_bil, target_tensor) + F.l1_loss(x_double_pr, target_tensor)

                # %%
                # ---- Motion Regularization ----
                loss_motion = 0.0

                for t in range(args.output_sequence_length):
                    vx = motion_[:, t, 0:1, :, :]   # [B,1,H,W]
                    vy = motion_[:, t, 1:2, :, :]

                    grad_vx_x = F.conv2d(vx, kx, padding=1)
                    grad_vx_y = F.conv2d(vx, ky, padding=1)
                    grad_vy_x = F.conv2d(vy, kx, padding=1)
                    grad_vy_y = F.conv2d(vy, ky, padding=1)

                    smoothness = grad_vx_x**2 + grad_vx_y**2 + grad_vy_x**2 + grad_vy_y**2
                    loss_motion += smoothness.mean()

                loss_motion = loss_motion / args.output_sequence_length

                # %%
                # ---- Total Loss ----
                lambda_motion = 1e-2
                loss = loss_accum + lambda_motion * loss_motion

            # %%
            # Backward + Optimizer Step (with AMP)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
            # %%
            # --- Logging ---
            if show_progress and (global_step % 100 == 0):
                train_bar.set_postfix(loss=loss.item(), accum=loss_accum.item(), motion=loss_motion.item())
            
            if global_step % 10000 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                wandb.log({
                    "train/loss": loss.item(),
                    "train/loss_accum": loss_accum.item(),
                    "train/loss_motion": loss_motion.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)
                ckpt_path = os.path.join(args.checkpoint_dir, f"evo_net_step_{global_step}.pt")
                save_model_checkpoint(model, optimizer, scheduler, global_step, ckpt_path)
            # also save latest
            if global_step % 1000 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                save_model_checkpoint(model, optimizer, scheduler, global_step, latest_ckpt_path)

# %%
if __name__ == "__main__":
    # %%
    def in_notebook() -> bool:
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except Exception:
            return False
    
    # === Defaults ===
    defaults = SimpleNamespace(
        img_size=(256,288),
    )

    # === Argument parsing ===
    parser = argparse.ArgumentParser(description="Training configuration for wind prediction model")
    parser.add_argument('--variable', type=str, default='i10fg', help='Input variable to predict')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='nowcastnet', help='Model to use')
    parser.add_argument('--activation_layer', type=str, default='gelu', help='Activation function')
    parser.add_argument('--transform', type=str_or_none, default=None, help='Data transformation type')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--weights_seed', type=int, default=42, help='Random seed for weight initialization')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--loss_name', type=str, default='MaskedCharbonnierLoss', help='Loss function name')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--input_sequence_length', type=int, default=36, help='Input window size (number of timesteps)')
    parser.add_argument('--output_sequence_length', type=int, default=36, help='Output window size (number of timesteps)')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for input/output windows')
    parser.add_argument('--forecast_offset', type=int, default=0, help='Offset for the forecast start time')
    parser.add_argument('--train_mode', type=int, default=0, help='Whether the model is in training mode: 0 for evolution net, 1 for nowcastnet')

    args_ns, _ = parser.parse_known_args([] if in_notebook() else None)

    # === Merge defaults + parsed ===
    merged = {**defaults.__dict__, **vars(args_ns)}
    args = SimpleNamespace(**merged)
    args.total_sequence_length = args.input_sequence_length + args.output_sequence_length

    # Checkpoint dir
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.model_name}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.loss_name}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.transform}"
    args.checkpoint_dir =  f"{args.checkpoint_dir}/in_window-{args.input_sequence_length}_out_window-{args.output_sequence_length}-step-{args.step_size}_offset-{args.forecast_offset}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ==================== Distributed setup ====================
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # Fallback: single GPU (non-DDP)
        print("Running without distributed setup (no torchrun detected)")
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device
    args.local_rank = local_rank
    def is_distributed():
        return dist.is_available() and dist.is_initialized()       # useful for checking if we are in a distributed environment
    
    # %%
    # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc')
    RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
    RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
    orography = orography.orog.values

    mask = xr.open_dataset('mask_2d.nc').mask
    mask_tensor = torch.tensor(mask.values, device=args.device)  # [H, W], defnitely send it to device
    
    # %%
    zarr_store = 'data/NYSM.zarr'
    train_val_dates_range = ['2019-01-01T00:00:00', '2019-12-31T23:59:59']
    test_dates_range = ['2023-03-26T02:00:00','2023-03-26T07:59:59']
    freq = '5min'
    data_seed = 42

    NYSM_stats = xr.open_dataset('NYSM_variable_stats.nc')
    input_stats = NYSM_stats.sel(variable=[args.variable])
    target_stats = NYSM_stats.sel(variable=[args.variable])
    # Standardization
    if args.transform is not None:
        input_transform = Transform(
            mode=args.transform,  # 'standard' or 'minmax'
            stats=input_stats,
            feature_axis=-1     # Channels last
        )   
        target_transform = Transform(
            mode=args.transform,  # 'standard' or 'minmax'
            stats=target_stats,
            feature_axis=-1     # Channels last
        )
    else:
        input_transform = None
        target_transform = None

    mode = 'train'
    train_dataset = nowcast_dataset(
        zarr_store,
        args.variable,
        train_val_dates_range,
        args.input_sequence_length,
        args.output_sequence_length,
        freq,
        missing_times=None,
        mode=mode,
        data_seed=data_seed,
        step_size=args.step_size,
        forecast_offset=args.forecast_offset
        )

    if is_distributed():
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # shuffle if not using DDP
        pin_memory=True,prefetch_factor=4, persistent_workers=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    mode = 'val'
    validation_dataset = nowcast_dataset(
        zarr_store,
        args.variable,
        train_val_dates_range,
        args.input_sequence_length,
        args.output_sequence_length,
        freq,
        missing_times=None,
        mode=mode,
        data_seed=data_seed,
        step_size=args.output_sequence_length,  # non-overlapping time-series in validation
        forecast_offset=args.forecast_offset
    )

    if is_distributed():
        validation_sampler = DistributedSampler(validation_dataset)
    else:
        validation_sampler = None
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        shuffle=(validation_sampler is None), # shuffle if not using DDP
        pin_memory=True,prefetch_factor=4, persistent_workers=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Data loaded successfully.")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")

    # %%
    # === Initializing the wandb ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.init(
            project="Gust_Nowcast",
            config={
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "input_sequence_length": args.input_sequence_length,
                "output_sequence_length": args.output_sequence_length,
                "train_val_dates_range": train_val_dates_range,
                "transform": args.transform,
            },
            name=args.checkpoint_dir[len('checkpoints/'):].replace('/','_'),
            dir="wandb_logs"
        )
    
    # %%
    if args.train_mode == 0:
        train_evolution_net(train_dataloader, train_sampler,args)
    
    # %%
    # === Barrier to ensure all ranks wait for checkpoint ===
    if dist.is_initialized():
        dist.barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Training and validation completed.")
    
    # === Finish run and destroy process group ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()