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

from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss, MaskedCombinedMAEQuantileLoss

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store, reshape_patch, reshape_patch_back

# === PredRNN ===#
predrnn_path = Path.cwd() / "external" / "predrnn"
sys.path.insert(0, str(predrnn_path))
from types import SimpleNamespace

# %%
def run_epochs(model, train_dataloader, val_dataloader, optimizer, criterion, metric,
               train_sampler, scheduler, early_stopping, mask_tensor,input_transform=None,target_transform=None,
               args=None):

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # === Optional resume ===
    start_epoch = 0
    best_val_loss = float("inf")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")

    if args.resume and os.path.exists(latest_ckpt_path):
        model, optimizer, scheduler, start_epoch = restore_model_checkpoint(model, optimizer, scheduler, latest_ckpt_path, args.device)

    itr = 0
    eta = args.sampling_start_value

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Mask tensor needs to be in same shape as frames_tensor
        mask_tensor_expanded = mask_tensor[None, None, :, :, None].to(args.device)  # [1,1,H,W,1]
        def forward_step(frames_tensor, real_input_flag):
            # Transform input and target
            if input_transform is not None:
                frames_tensor = input_transform(frames_tensor)

            # Mask frames with NYS boundary
            masked_frames_tensor = torch.where(mask_tensor_expanded, frames_tensor, 0)

            # Patchify the frames_tensor and real_input_flags
            patched_frames_tensor = reshape_patch(masked_frames_tensor, args.patch_size)
            real_input_flag = reshape_patch(real_input_flag, args.patch_size)
            
            next_frames, decouple_loss = model(patched_frames_tensor, real_input_flag)

            # Apply inverse patching
            next_frames = reshape_patch_back(next_frames, args.patch_size)

            # Mask next frames with NYS boundary
            masked_next_frames = torch.where(mask_tensor_expanded, next_frames, 0)

            loss = criterion(masked_next_frames, masked_frames_tensor[:, 1:]) + decouple_loss

            # Compute the metric, ONLY ON THE PREDICTION HORIZON, on inverse transformed, if needed
            if input_transform is not None:
                frames_tensor = input_transform.inverse(frames_tensor)
                # Mask frames with NYS boundary
                masked_frames_tensor = torch.where(mask_tensor_expanded, frames_tensor, 0)

                next_frames = input_transform.inverse(next_frames)
                masked_next_frames = torch.where(mask_tensor_expanded, next_frames, 0)

            metric_value = metric(masked_next_frames[:,args.input_window_size-1:], masked_frames_tensor[:,args.input_window_size:])

            return loss, metric_value

        # === Training ===
        model.train()
        train_loss_total = 0.0
        train_metric_total = 0.0
        show_progress = not dist.is_initialized() or dist.get_rank() == 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", leave=False) if show_progress else train_dataloader
        for batch_idx,batch in enumerate(train_bar):
            #start_time = time.time()
            itr += 1

            input_tensor, target_tensor,_,_ = batch
            frames_tensor = torch.cat((input_tensor, target_tensor), dim=1)  # [B, Tin+Tout, H, W]
            frames_tensor = frames_tensor.unsqueeze(-1)     # [B, Tin+Tout, H, W, 1]

            frames_tensor = frames_tensor.to(args.device, non_blocking=True)   # [B, Tin+Tout, H, W, 1]

            if args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(itr,args)
            else:
                eta, real_input_flag = schedule_sampling(eta, itr,args)

            optimizer.zero_grad()

            loss, metric_value = forward_step(frames_tensor, real_input_flag)

            if args.reverse_input:
                frames_tensor_rev = torch.flip(frames_tensor, dims=[1])
                loss_rev, _ = forward_step(frames_tensor_rev, real_input_flag)
                loss = (loss + loss_rev) / 2

            loss.backward()
            optimizer.step()

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
                input_tensor, target_tensor,_,_ = batch
                frames_tensor = torch.cat((input_tensor, target_tensor), dim=1)  # [B, Tin+Tout, H, W]
                frames_tensor = frames_tensor.unsqueeze(-1)     # [B, Tin+Tout, H, W, 1]

                frames_tensor = frames_tensor.to(args.device, non_blocking=True)   # [B, Tin+Tout, H, W, 1]

                _, real_input_flag = flags_eval(args)

                loss, metric_value = forward_step(frames_tensor, real_input_flag)

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

# %%
if __name__ == "__main__":
    # %%
    # === Defaults ===
    defaults = SimpleNamespace(
        is_training=1,
        reverse_input=1,
        img_size=(256,288),
        img_channel=1,
        num_hidden="128,128,128,128",
        filter_size=5,
        stride=1,
        patch_size=(4,4),
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
    parser.add_argument('--model_name', type=str, default='predrnn_v2', help='Model to use')
    parser.add_argument('--activation_layer', type=str, default='gelu', help='Activation function')
    parser.add_argument('--transform', type=str_or_none, default=None, help='Data transformation type')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--weights_seed', type=int, default=42, help='Random seed for weight initialization')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--loss_name', type=str, default='MaskedCharbonnierLoss', help='Loss function name')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--input_window_size', type=int, default=36, help='Input window size (number of timesteps)')
    parser.add_argument('--output_window_size', type=int, default=36, help='Output window size (number of timesteps)')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for input/output windows')
    parser.add_argument('--forecast_offset', type=int, default=0, help='Offset for the forecast start time')
    args, unknown = parser.parse_known_args()   
    args.total_window_size = args.input_window_size + args.output_window_size
    # === Merge them both ===
    merged = {**vars(defaults), **vars(args)} 

    # Back to a single name space ===
    args = SimpleNamespace(**merged)

    # Checkpoint dir
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.model_name}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.loss_name}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{args.transform}"
    args.checkpoint_dir =  f"{args.checkpoint_dir}/in_window-{args.input_window_size}_out_window-{args.output_window_size}-step-{args.step_size}_offset-{args.forecast_offset}"
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
    # === Defining schedule sampling functions ===
    @torch.no_grad()
    def reserve_schedule_sampling_exp(itr, args, dtype=torch.float32):
        """
        Reverse schedule sampling flags in GRID space.
        Returns: real_input_flag of shape [B, T-2, H, W, C] (float32 on device)
        """
        B = args.batch_size
        Tin = args.input_window_size
        T   = args.total_window_size
        H, W = args.img_size
        C = args.img_channel
        device = args.device

        # r_eta / eta
        if itr < args.r_sampling_step_1:
            r_eta = 0.5
        elif itr < args.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * math.exp(-(itr - args.r_sampling_step_1) / float(args.r_exp_alpha))
        else:
            r_eta = 1.0

        if itr < args.r_sampling_step_1:
            eta = 0.5
        elif itr < args.r_sampling_step_2:
            eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
        else:
            eta = 0.0

        # random flips → booleans
        gen = None  # or pass a torch.Generator for reproducibility
        r_true_token = (torch.rand(B, Tin - 1, device=device, generator=gen) < r_eta)
        true_token   = (torch.rand(B, T - Tin - 1, device=device, generator=gen) < eta)

        # build flags list per-sample/time (keep vectorized over HWC)
        ones  = torch.ones (H, W, C, device=device, dtype=dtype)
        zeros = torch.zeros(H, W, C, device=device, dtype=dtype)

        # length = T-2
        flags = []
        for i in range(B):
            row = []
            # j = 0 .. T-3  → split at Tin-1
            for j in range(T - 2):
                if j < Tin - 1:
                    row.append(ones if r_true_token[i, j] else zeros)
                else:
                    jj = j - (Tin - 1)
                    row.append(ones if true_token[i, jj] else zeros)
            flags.append(torch.stack(row, dim=0))   # [T-2,H,W,C]
        real_input_flag = torch.stack(flags, dim=0) # [B,T-2,H,W,C]
        return real_input_flag  # dtype float32


    @torch.no_grad()
    def schedule_sampling(eta, itr, args, dtype=torch.float32):
        """
        Forward schedule sampling flags in GRID space.
        Returns: (eta_new, real_input_flag) with shape [B, T-Tin-1, H, W, C]
        """
        B = args.batch_size
        Tin = args.input_window_size
        T   = args.total_window_size
        H, W = args.img_size
        C = args.img_channel
        device = args.device

        if not getattr(args, "scheduled_sampling", True):
            # zeros like original
            rif = torch.zeros(B, T - Tin - 1, H, W, C, device=device, dtype=dtype)
            return 0.0, rif

        # update eta
        if itr < args.sampling_stop_iter:
            eta_new = eta - args.sampling_changing_rate
        else:
            eta_new = 0.0

        # Bernoulli draws
        gen = None
        true_token = (torch.rand(B, T - Tin - 1, device=device, generator=gen) < eta_new)

        ones  = torch.ones (H, W, C, device=device, dtype=dtype)
        zeros = torch.zeros(H, W, C, device=device, dtype=dtype)

        flags = []
        for i in range(B):
            row = [ones if true_token[i, j] else zeros for j in range(T - Tin - 1)]
            flags.append(torch.stack(row, dim=0))   # [T-Tin-1,H,W,C]
        real_input_flag = torch.stack(flags, dim=0) # [B,T-Tin-1,H,W,C]
        return eta_new, real_input_flag
    
    @torch.no_grad()
    def flags_eval(args):
        B      = args.batch_size
        Tin    = args.input_window_size
        T      = args.total_window_size
        H, W   = args.img_size
        C      = args.img_channel
        device = args.device
        dtype  = torch.float32

        if args.reverse_scheduled_sampling == 1:
            L = T - 2  # reverse mode length
            # timeline index j=0..T-3 corresponds to decisions for times 1..T-2
            # set first Tin-1 to 1 (teacher forcing), rest 0
            ones  = torch.ones (B, Tin-1, H, W, C, device=device, dtype=dtype)
            zeros = torch.zeros(B, L-(Tin-1), H, W, C, device=device, dtype=dtype)
            flags = torch.cat([ones, zeros], dim=1)  # [B,T-2,H,W,C]
            eta   = None
            return eta, flags
        else:
            L = T - Tin - 1  # forward mode length
            flags = torch.zeros(B, L, H, W, C, device=device, dtype=dtype)  # pure open-loop
            eta   = None
            return eta, flags
    
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
        args.input_window_size,
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
        args.input_window_size,
        args.output_window_size,
        freq,
        missing_times=None,
        mode=mode,
        data_seed=data_seed,
        step_size=args.step_size,
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
    num_hidden = [int(x) for x in args.num_hidden.split(',')]
    num_layers = len(num_hidden)
    model = predrnn_v2.RNN(num_layers,num_hidden,args).to(args.device)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define the loss criterion and metric here, based on input loss name. The functions are sent to the GPU inside
    if args.loss_name == "MaskedMSELoss":
        criterion = MaskedMSELoss(mask_tensor)
    elif args.loss_name == "MaskedRMSELoss":
        criterion = MaskedRMSELoss(mask_tensor)
    elif args.loss_name == "MaskedTVLoss":
        criterion = MaskedTVLoss(mask_tensor,tv_loss_weight=0.001, beta=0.5)    
    elif args.loss_name == "MaskedCharbonnierLoss":
        criterion = MaskedCharbonnierLoss(mask_tensor,eps=1e-3)
    elif args.loss_name == "MaskedCombinedMAEQuantileLoss":
        criterion = MaskedCombinedMAEQuantileLoss(mask_tensor, tau=0.95, mae_weight=0.5, quantile_weight=0.5)
    
    metric = MaskedRMSELoss(mask_tensor)

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
            config={
                "model_name": args.model_name,
                "activation_layer": args.activation_layer,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "weights_seed": args.weights_seed,
                "loss_name": args.loss_name,
                "input_window_size": args.input_window_size,
                "output_window_size": args.output_window_size,
                "train_val_dates_range": train_val_dates_range,
                "transform": args.transform,
            },
            name=args.checkpoint_dir[len('checkpoints/'):].replace('/','_'),
            dir="wandb_logs"
        )

    # %%
    # === Run the training and validation ===
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
        mask_tensor=mask_tensor,
        input_transform=input_transform,
        target_transform=target_transform,
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