# === Top level imports ===
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR


import xarray as xr
import zarr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time

import os
import wandb, argparse, sys
from tqdm import tqdm

from models.Google_Unet import GoogleUNet
from models.Deep_CNN import DCNN
from models.UNet import UNet
from models.SwinT2_UNet import SwinT2UNet
from models.util import initialize_weights_xavier,initialize_weights_he

# %%
def str_or_none(v):
    return None if v.lower() == 'none' else v

def int_or_none(v):
    return None if v.lower() == 'none' else int(v)

def bool_from_str(v):
    return v.lower() == 'true'
# %%
# === Early stopping, and checkpointing functions ===
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
def save_model_checkpoint(model, optimizer,scheduler, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print(f" Model checkpoint saved at: {path}")

def strip_ddp_prefix(state_dict):
    """Remove 'module.' prefix if loading DDP model into non-DDP."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def restore_model_checkpoint(model, optimizer=None, scheduler=None, path=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try stripping 'module.' prefix
        print("Warning: DDP prefix found, attempting to strip 'module.' from keys...")
        state_dict = strip_ddp_prefix(state_dict)
        model.load_state_dict(state_dict)
    # Only load optimizer if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Only load scheduler if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Restored checkpoint from: {path} (epoch {checkpoint['epoch']})")
    return model, optimizer, scheduler, start_epoch

# %%
# === Computing the outputs on test data and saving them to zarr ===
def init_zarr_store(zarr_store, dates,variable):
    orography = xr.open_dataset('orography.nc')
    orography.attrs = {}
    template = xr.full_like(orography.orog.expand_dims(time=dates),fill_value=np.nan,dtype='float32')
    template['time'] = dates
    template = template.chunk({'time': 24})
    template = template.transpose('time','y','x')
    template = template.assign_coords({
        'latitude': orography.latitude,
        'longitude': orography.longitude
    })
    template.to_dataset(name = variable).to_zarr(zarr_store, compute=False, mode='w')

# === Reshaping functions ===
def reshape_patch(img_tensor: torch.Tensor, patch_size):
    """
    img_tensor: [B, T, C, H, W]
    patch_size: (ph, pw)
    Returns: [B, T, C*ph*pw, H/ph, W/pw]
    """
    assert img_tensor.ndim == 5
    B, T, C, H, W = img_tensor.shape
    ph, pw = patch_size
    assert H % ph == 0 and W % pw == 0, "H,W must be divisible by patch_size"

    # [B,T,C,H/ph,ph,W/pw,pw]
    x = img_tensor.view(B, T, C, H // ph, ph, W // pw, pw)
    # [B,T,C,H/ph,W/pw,ph,pw]
    x = x.permute(0, 1, 2, 4, 6, 3, 5).contiguous()
    # [B,T,C,H/ph,W/pw,ph*pw]
    patch_tensor = x.view(B, T, C*ph * pw, H // ph, W // pw)
    return patch_tensor

def reshape_patch_back(patch_tensor: torch.Tensor, patch_size):
    """
    patch_tensor: [B, T, C*ph*pw, H/ph, W/pw]
    patch_size: (ph, pw)
    Returns: [B, T, C, H, W]
    """
    assert patch_tensor.ndim == 5
    B, T, CC, Hh, Ww = patch_tensor.shape
    ph, pw = patch_size
    C = CC // (ph * pw)

    # [B,T,C,ph,pw,Hh,Ww]
    x = patch_tensor.view(B, T, C, ph, pw, Hh, Ww)
    # [B,T,C,Hh,ph,Ww,pw]
    x = x.permute(0, 1, 2, 5, 3, 6,4).contiguous()
    # [B,T,C,Hh*ph,Ww*pw]
    img_tensor = x.view(B, T, C, Hh * ph, Ww * pw)
    return img_tensor

def wandb_safe_config(args):
    safe = {}
    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool, list, tuple, dict)) or v is None:
            safe[k] = v
        elif isinstance(v, torch.device):
            safe[k] = str(v)   # e.g. "cuda:0"
        else:
            safe[k] = str(v)   # fallback: store repr
    return safe