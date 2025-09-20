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
from models.CGsNet import CGsNet

from losses import MaskedErrorLoss, MaskedCharbonnierLoss, MaskedCombinedMAEQuantileLoss
from types import SimpleNamespace

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store, reshape_patch, reshape_patch_back, wandb_safe_config

# %%
def forward_step(input_tensor,target_tensor,use_teacher_forcing,
                 model, criterion, metric,
                 mask_tensor_expanded, args,
                 input_transform=None, return_preds=False):
    """
    One forward + loss + metric computation step.
    
    Parameters
    ----------
    frames_tensor : torch.Tensor
        Shape [B, Tin+Tout, H, W, 1]
    model : nn.Module
        CGsNet model
    criterion : callable
        Loss function
    metric : callable
        Metric function
    mask_tensor_expanded : torch.Tensor
        Mask of shape [1,1,H,W,1]
    args : Namespace
        Holds config (patch_size, input_sequence_length, etc.)
    input_transform : Transform, optional
        Transformation with .__call__ and .inverse
    return_preds : bool, optional
        Whether to return predictions, only for testing
    """
    # Transform input
    if input_transform is not None:
        input_tensor = input_transform(input_tensor)
        target_tensor = input_transform(target_tensor)

    # Mask input
    masked_input_tensor = torch.where(mask_tensor_expanded, input_tensor, 0)
    masked_target_tensor = torch.where(mask_tensor_expanded, target_tensor, 0)

    # Forward model
    next_frames = model(masked_input_tensor, None,use_teacher_forcing)  #For now, the target_tensor and use_teacher_forcing are not provided. 

    # Mask predictions
    masked_next_frames = torch.where(mask_tensor_expanded, next_frames, 0)

    # Compute loss
    if criterion is not None:
        loss = criterion(masked_next_frames, masked_target_tensor)

    # Metric on inverse-transformed data if needed
    if input_transform is not None:
        target_tensor = input_transform.inverse(target_tensor)
        masked_target_tensor = torch.where(mask_tensor_expanded, target_tensor, 0)

        next_frames = input_transform.inverse(next_frames)
        masked_next_frames = torch.where(mask_tensor_expanded, next_frames, 0)

    if metric is not None:
        metric_value = metric(
            masked_next_frames,
            masked_target_tensor,
            mode='mse', reduction='mean'
        )

    if (criterion is not None) and (metric is not None):
        if return_preds:
            return loss, metric_value, masked_next_frames
        else:
            return loss, metric_value

    elif (criterion is not None) and (metric is None):
        if return_preds:
            return loss, masked_next_frames
        else:
            return loss

    elif (criterion is None) and (metric is not None):
        if return_preds:
            return metric_value, masked_next_frames
        else:
            return metric_value

    else:  # criterion is None and metric is None
        if return_preds:
            return masked_next_frames
        else:
            return None  # or: raise ValueError("Nothing to return")

def run_epochs(model, train_dataloader, val_dataloader, optimizer, criterion, metric,
               train_sampler, scheduler, early_stopping, mask_tensor_expanded, input_transform=None,target_transform=None,
               args=None):

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # === Optional resume ===
    start_epoch = 0
    best_val_loss = float("inf")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")

    if args.resume and os.path.exists(latest_ckpt_path):
        model, optimizer, scheduler, start_epoch = restore_model_checkpoint(model, optimizer, scheduler, latest_ckpt_path, args.device)

    itr = 0

    scaler = GradScaler("cuda")   # <<< AMP scaler

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # === Training ===
        model.train()
        train_loss_total = 0.0
        train_metric_total = 0.0
        show_progress = not dist.is_initialized() or dist.get_rank() == 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", leave=False) if show_progress else train_dataloader
        for batch_idx,batch in enumerate(train_bar):
            #start_time = time.time()
            itr += 1

            input_tensor, target_tensor,_,_ = batch  # input_tensor, target_tensor: [B, Tin, H, W], [B, Tout, H, W]
            input_tensor = input_tensor.unsqueeze(2)     # [B, Tin+Tout, 1, H, W]
            target_tensor = target_tensor.unsqueeze(2)   # [B, Tin+Tout, 1, H, W]

            input_tensor = input_tensor.to(args.device, non_blocking=True)   # [B, Tin+Tout, 1, H, W]
            target_tensor = target_tensor.to(args.device, non_blocking=True) # [B, Tin+Tout, 1, H, W]

            Bcur = input_tensor.size(0)

            optimizer.zero_grad(set_to_none=True)
            
            use_teacher_forcing = False

            # === AMP forward/backward ===
            with autocast("cuda", dtype=torch.float16):   # <<< AMP autocast
                loss, metric_value  = forward_step(input_tensor, target_tensor, use_teacher_forcing, model, criterion, metric, mask_tensor_expanded, args, input_transform, return_preds=False)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            train_metric_total += metric_value.item()

            if show_progress:
                train_bar.set_postfix(loss=loss.item(), metric=metric_value.item())
            
            # end_time = time.time()
            # print(f"[Batch {batch_idx}] Data load time: {end_time - start_time:.4f} seconds")
            # #Break early to test
            # if batch_idx == 5:
            #    break

        avg_train_loss = train_loss_total / len(train_dataloader)
        avg_train_metric = train_metric_total / len(train_dataloader)

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        val_metric_total = 0.0
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]", leave=False) if show_progress else val_dataloader
        with torch.no_grad():
            for batch in val_bar:
                input_tensor, target_tensor,_,_ = batch # input_tensor, target_tensor: [B, Tin, H, W], [B, Tout, H, W]
                input_tensor = input_tensor.unsqueeze(2)     # [B, Tin+Tout, 1, H, W]
                target_tensor = target_tensor.unsqueeze(2)   # [B, Tin+Tout, 1, H, W]

                input_tensor = input_tensor.to(args.device, non_blocking=True)   # [B, Tin+Tout, 1, H, W]
                target_tensor = target_tensor.to(args.device, non_blocking=True) # [B, Tin+Tout, 1, H, W]

                with autocast("cuda", dtype=torch.float16):   # AMP works for eval too
                    loss, metric_value  = forward_step(input_tensor, target_tensor, False, model, criterion, metric, mask_tensor_expanded, args, input_transform, return_preds=False)

                val_loss_total += loss.item()
                val_metric_total += metric_value.item()

                if show_progress:
                    val_bar.set_postfix(loss=loss.item(), metric=metric_value.item())
        avg_val_loss = val_loss_total / len(val_dataloader)
        avg_val_metric = val_metric_total / len(val_dataloader)

        # === Scheduler step ===
        scheduler.step()

        # === Log to Weights & Biases ===
        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_metric": avg_train_metric,
                "val_loss": avg_val_loss,
                "val_metric": avg_val_metric,
                "learning_rate": scheduler.get_last_lr()[0],   # log current LR
            })

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Epoch {epoch+1}]  Train Loss: {avg_train_loss:.4f} |  Val Loss: {avg_val_loss:.4f}  | LR: {scheduler.get_last_lr()[0]:.6f}")

        # === Save Checkpoints ===
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Update best model if needed
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                save_model_checkpoint(model, optimizer, scheduler, epoch, best_ckpt_path)

            # Always update latest checkpoint
            save_model_checkpoint(model, optimizer, scheduler, epoch, latest_ckpt_path)

        # === Early stopping check (in ALL ranks, after validation step) ===
        if early_stopping is not None:
            stop_flag = torch.tensor(
                int(early_stopping(avg_val_loss)), device=args.device, dtype=torch.int
            )
            if dist.is_initialized():
                dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
            if stop_flag.item() > 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                break

@torch.no_grad()
def run_inference(model, test_dataloader, mask_tensor_expanded, criterion, metric, args, input_transform=None):
    # load the best model Handle DDP 'module.' prefix
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    model, _, _, _ = restore_model_checkpoint(model, None, None, best_ckpt_path, args.device)

    model.eval()
    total_loss, total_metric = 0.0, 0.0
    preds_all = []

    for batch in tqdm(test_dataloader, desc="[Test]"):
        input_tensor, target_tensor,_,_ = batch  # input_tensor, target_tensor: [B, Tin, H, W], [B, Tout, H, W]
        input_tensor = input_tensor.unsqueeze(2)     # [B, Tin+Tout, 1, H, W]
        target_tensor = target_tensor.unsqueeze(2)   # [B, Tin+Tout, 1, H, W]

        input_tensor = input_tensor.to(args.device, non_blocking=True)   # [B, Tin+Tout, 1, H, W]
        target_tensor = target_tensor.to(args.device, non_blocking=True) # [B, Tin+Tout, 1, H, W]

        loss, metric_value, preds = forward_step(
            input_tensor, target_tensor, False,
            model, criterion, metric,
            mask_tensor_expanded, args,
            input_transform, return_preds=True
        )
        total_loss += loss.item()
        total_metric += metric_value.item()
        preds_all.append(preds[:, args.input_sequence_length-1 : ].cpu())

    avg_loss   = total_loss / len(test_dataloader)
    avg_metric = total_metric / len(test_dataloader)
    return avg_loss, avg_metric, torch.cat(preds_all, dim=0)
# %%
if __name__ == "__main__":
    # %%
    # === Defaults ===
    defaults = SimpleNamespace(
        height=256,
        width=288,
        downscale_factor=4,
        num_hidden=(128,128,128,128),
        num_channels_in=1,
        num_channels_out=1,
        cnn_hidden_size=64,
        rnn_input_dim=64,
        phycell_hidden_dims=[64],
        kernel_size_phycell=3,
        convlstm_hidden_dims=[64],
        kernel_size_convlstm=3,
        input_sequence_length=36,
        output_sequence_length=36
    )

    # === Argument parsing ===
    parser = argparse.ArgumentParser(description="Training configuration for wind prediction model")
    parser.add_argument('--variable', type=str, default='i10fg', help='Input variable to predict')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='CGsNet', help='Model to use')
    parser.add_argument('--transform', type=str_or_none, default='minmax', help='Data transformation type')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--loss_name', type=str, default='MaskedCharbonnierLoss', help='Loss function name')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--input_sequence_length', type=int, default=36, help='Input window size (number of timesteps)')
    parser.add_argument('--output_sequence_length', type=int, default=36, help='Output window size (number of timesteps)')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for input/output windows')
    parser.add_argument('--forecast_offset', type=int, default=0, help='Offset for the forecast start time')
    parser.add_argument('--is_training', type=int, default=1, help='Whether the model is in training mode')
    args, unknown = parser.parse_known_args()   
    args.total_sequence_length = args.input_sequence_length + args.output_sequence_length
    # === Merge them both ===
    merged = {**vars(defaults), **vars(args)}
    # Back to a single name space ===
    args = SimpleNamespace(**merged)

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
    mask_tensor_expanded = mask_tensor[None, None, None, :, :].to(args.device)  # [1,1,1, H,W]  # This is essential in masking and loss computation. Care must be taken for the shape according to the model design.
    
    # %%
    zarr_store = 'data/NYSM.zarr'
    train_val_dates_range = ['2021-01-01T00:00:00', '2023-12-31T23:59:59']
    test_dates_range = ['2024-01-01T00:00:00','2024-12-31T23:59:59']
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
            feature_axis=2     # Channels 2, in B,T,C,H,W
        )   
        target_transform = Transform(
            mode=args.transform,  # 'standard' or 'minmax'
            stats=target_stats,
            feature_axis=2     # Channels 2, in B,T,C,H,W
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
    # === Set up device, model, loss, optimizer ===
    model = CGsNet(args).to(args.device)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define the loss criterion and metric here, based on input loss name. The functions are sent to the GPU inside 
    if args.loss_name == "MaskedCharbonnierLoss":
        criterion = MaskedCharbonnierLoss(mask_tensor_expanded,eps=1e-3)
    elif args.loss_name == "MaskedCombinedMAEQuantileLoss":
        criterion = MaskedCombinedMAEQuantileLoss(mask_tensor_expanded, tau=0.95, mae_weight=0.5, quantile_weight=0.5)

    metric = MaskedErrorLoss(mask_tensor_expanded).to(args.device)
    # === Optimizer, scheduler, and early stopping ===
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    early_stopping = EarlyStopping(patience=20, min_delta=0.0)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Model created and moved to device.")

    # %%
    # === Initializing the wandb ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.init(
            project="Gust_Nowcast",
            config=wandb_safe_config(args),  # Use the safe config function to avoid issues with non-serializable types
            name=args.checkpoint_dir[len('checkpoints/'):].replace('/','_'),
            dir="wandb_logs"
        )

    # %%
    # === Run the training and validation ===
    if args.is_training:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Starting training and validation...")
        run_epochs(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=validation_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            train_sampler=train_sampler, 
            scheduler=scheduler,
            early_stopping=early_stopping,
            mask_tensor_expanded=mask_tensor_expanded,
            input_transform=input_transform,
            target_transform=target_transform,
            args=args,
        )
    else:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Starting testing...")
        test_dataset = nowcast_dataset(
        zarr_store,
        args.variable,
        test_dates_range,
        args.input_sequence_length,
        args.output_sequence_length,
        freq,
        missing_times=None,
        mode='test',
        data_seed=data_seed,
        step_size=args.output_sequence_length,  # non-overlapinput_tensorping time-series in validation
        forecast_offset=args.forecast_offset
        )

        if is_distributed():
            test_sampler = DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=(test_sampler is None), # shuffle if not using DDP
            pin_memory=True,prefetch_factor=4, persistent_workers=True,
            num_workers=args.num_workers,
            drop_last=False
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Data loaded successfully.")
            print(f"Test dataset size: {len(test_dataset)}")
        _, _, ds = run_inference(
            model=model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            metric=metric,
            mask_tensor_expanded=mask_tensor_expanded,
            input_transform=input_transform,
            args=args,
        )
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