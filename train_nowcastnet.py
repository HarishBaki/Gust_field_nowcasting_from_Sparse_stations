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

from losses import MaskedErrorLoss, MaskedTVLoss, MaskedCharbonnierLoss, MaskedCombinedMAEQuantileLoss

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store, reshape_patch, reshape_patch_back
from types import SimpleNamespace
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
        reverse_input=1,
        img_size=(256,288),
        img_channel=1,
        num_hidden="128,128,128,128",
        filter_size=5,
        stride=1,
        patch_size=(8,8),
        layer_norm=0,
        decouple_beta=0.1,
        reverse_scheduled_sampling=1,
        r_sampling_step_1=25000,
        r_sampling_step_2=50000,
        r_exp_alpha=2500,
        sampling_start_value=1,
        lr=0.0001,
        max_iterations=80000,
        display_interval=100,
        test_interval=5000,
        snapshot_interval=5000,
        pretrained_model=None,  # optional
        injection_action='concat',
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
    parser.add_argument('--output_window_size', type=int, default=36, help='Output window size (number of timesteps)')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for input/output windows')
    parser.add_argument('--forecast_offset', type=int, default=0, help='Offset for the forecast start time')
    parser.add_argument('--is_training', type=int, default=0, help='Whether the model is in training mode')

    args_ns, _ = parser.parse_known_args([] if in_notebook() else None)

    # === Merge defaults + parsed ===
    merged = {**defaults.__dict__, **vars(args_ns)}
    args = SimpleNamespace(**merged)
    args.total_window_size = args.input_sequence_length + args.output_window_size

    # Checkpoint dir
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.model_name}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.loss_name}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.transform}"
    args.checkpoint_dir =  f"{args.checkpoint_dir}/in_window-{args.input_sequence_length}_out_window-{args.output_window_size}-step-{args.step_size}_offset-{args.forecast_offset}"
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
        args.output_window_size,
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
        args.output_window_size,
        freq,
        missing_times=None,
        mode=mode,
        data_seed=data_seed,
        step_size=args.output_window_size,  # non-overlapping time-series in validation
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
    batch = next(iter(train_dataloader),None)
    input_tensor, target_tensor,_,_ = batch

    # %%
    # === Set up device, model, loss, optimizer ===
    num_hidden = [int(x) for x in args.num_hidden.split(',')]
    num_layers = len(num_hidden)
    model = predrnn_v2.RNN(num_layers,num_hidden,args).to(args.device)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])