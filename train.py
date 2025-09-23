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
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time

import os
import wandb, argparse, sys
from tqdm import tqdm

from data_loader import nowcast_dataset, Transform
from models.Google_Unet import GoogleUNet
from models.Deep_CNN import DCNN
from models.UNet import UNet
from models.SwinT2_UNet import SwinT2UNet
from models.util import initialize_weights_xavier,initialize_weights_he

from losses import MaskedErrorLoss, MaskedCharbonnierLoss, MaskedCombinedMAEQuantileLoss

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store

# %%
def run_epochs(model, train_dataloader, val_dataloader, optimizer, criterion, metric, device, num_epochs,
               checkpoint_dir, train_sampler, scheduler, early_stopping, mask_tensor_expanded, input_transform=None, target_transform=None, resume=False):

    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Optional resume ===
    start_epoch = 0
    best_val_loss = float("inf")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")

    if resume and os.path.exists(latest_ckpt_path):
        model, optimizer, scheduler, start_epoch = restore_model_checkpoint(model, optimizer, scheduler, latest_ckpt_path, device)

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # === Training ===
        model.train()
        train_loss_total = 0.0
        train_metric_total = 0.0
        show_progress = not dist.is_initialized() or dist.get_rank() == 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False) if show_progress else train_dataloader
        for batch_idx,batch in enumerate(train_bar):
            #start_time = time.time()
            input_tensor, target_tensor,_,_ = batch
            input_tensor = input_tensor.to(device, non_blocking=True)   # [B, C, H, W]
            target_tensor = target_tensor.to(device, non_blocking=True)    # [B, C, H, W]

            # Transform input and target
            if input_transform is not None:
                input_tensor = input_transform(input_tensor)
                target_tensor = target_transform(target_tensor)

            #end_time = time.time()
            #print(f"[Batch {batch_idx}] Data load time: {end_time - start_time:.4f} seconds")
            # Break early to test
            #if batch_idx == 5:
            #    break

            optimizer.zero_grad()
            # === AMP forward/backward ===
            with autocast("cuda", dtype=torch.bfloat16):   # <<< AMP autocast
                output = model(torch.where(mask_tensor_expanded, input_tensor, 0))
                loss = criterion(torch.where(mask_tensor_expanded, output, 0), torch.where(mask_tensor_expanded, target_tensor, 0))
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            # === Optional: Apply inverse transform if needed ===
            if target_transform is not None:
                output = target_transform.inverse(output)
                target_tensor = target_transform.inverse(target_tensor)

            # Compute the metric
            metric_value = metric(torch.where(mask_tensor_expanded, output, 0), torch.where(mask_tensor_expanded, target_tensor, 0),mode='mse', reduction='mean')
            train_metric_total += metric_value.item()

            if show_progress:
                train_bar.set_postfix(loss=loss.item(), metric=metric_value.item())

        avg_train_loss = train_loss_total / len(train_dataloader)
        avg_train_metric = train_metric_total / len(train_dataloader)

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        val_metric_total = 0.0
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False) if show_progress else val_dataloader
        with torch.no_grad():
            for batch in val_bar:
                input_tensor, target_tensor,_,_ = batch
                input_tensor = input_tensor.to(device, non_blocking=True)
                target_tensor = target_tensor.to(device, non_blocking=True)

                # Transform input and target
                if input_transform is not None:
                    input_tensor = input_transform(input_tensor)
                    target_tensor = target_transform(target_tensor)

                output = model(torch.where(mask_tensor_expanded, input_tensor, 0))
                loss = criterion(torch.where(mask_tensor_expanded, output, 0), torch.where(mask_tensor_expanded, target_tensor, 0))
                val_loss_total += loss.item()

                # === Optional: Apply inverse transform if needed ===
                if target_transform is not None:
                    output = target_transform.inverse(output)
                    target_tensor = target_transform.inverse(target_tensor)
                
                # Compute the metric
                metric_value = metric(torch.where(mask_tensor_expanded, output, 0), torch.where(mask_tensor_expanded, target_tensor, 0),mode='mse', reduction='mean')
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
                best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_model_checkpoint(model, optimizer, scheduler, epoch, best_ckpt_path)

            # Always update latest checkpoint
            save_model_checkpoint(model, optimizer, scheduler, epoch, latest_ckpt_path)

        # === Early stopping check (in ALL ranks, after validation step) ===
        if early_stopping is not None:
            stop_flag = torch.tensor(
                int(early_stopping(avg_val_loss)), device=device, dtype=torch.int
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
    # === Argument parsing ===
    parser = argparse.ArgumentParser(description="Training configuration for wind prediction model")
    parser.add_argument('--variable', type=str, default='i10fg', help='Input variable to predict')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='SwinT2UNet', choices=['DCNN', 'UNet', 'SwinT2UNet', 'GoogleUNet'], help='Model to use')
    parser.add_argument('--activation_layer', type=str, default='gelu', help='Activation function')
    parser.add_argument('--transform', type=str_or_none, default='minmax', choices=['standard', 'minmax',None], help='Data transformation type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--weights_seed', type=int, default=42, help='Random seed for weight initialization')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--loss_name', type=str, default='MaskedCharbonnierLoss', help='Loss function name')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--input_sequence_length', type=int, default=36, help='Input window size (number of timesteps)')
    parser.add_argument('--output_sequence_length', type=int, default=36, help='Output window size (number of timesteps)')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for input/output windows')
    parser.add_argument('--forecast_offset', type=int, default=0, help='Offset for the forecast start time')
    args, unknown = parser.parse_known_args([] if 'ipykernel_launcher' in sys.argv[0] else None)

    # Update variables from parsed arguments
    variable = args.variable
    checkpoint_dir = args.checkpoint_dir
    model_name = args.model_name
    activation_layer = args.activation_layer
    transform = args.transform
    batch_size = args.batch_size
    num_workers = args.num_workers
    weights_seed = args.weights_seed
    num_epochs = args.num_epochs
    loss_name = args.loss_name
    resume = args.resume
    input_sequence_length = args.input_sequence_length
    output_sequence_length = args.output_sequence_length
    step_size = args.step_size
    forecast_offset = args.forecast_offset

    checkpoint_dir = f"{checkpoint_dir}/{model_name}"
    checkpoint_dir = f"{checkpoint_dir}/{loss_name}"
    checkpoint_dir = f"{checkpoint_dir}/{transform}"
    checkpoint_dir = f"{checkpoint_dir}/{activation_layer}-{weights_seed}"
    checkpoint_dir =  f"{checkpoint_dir}/in_window-{input_sequence_length}_out_window-{output_sequence_length}-step-{step_size}_offset-{forecast_offset}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # %%
    # ==================== Distributed setup ====================
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        # Get rank and set device as before
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # Fallback: single GPU (non-DDP) for debugging or interactive use
        print("Running without distributed setup (no torchrun detected)")
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def is_distributed():
        return dist.is_available() and dist.is_initialized()       # useful for checking if we are in a distributed environment

    # %%
    # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc')
    RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
    RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
    orography = orography.orog.values

    mask = xr.open_dataset('mask_2d.nc').mask
    mask_tensor = torch.tensor(mask.values, device=device)  # [H, W], defnitely send it to device
    mask_tensor_expanded = mask_tensor[None, None, :, :].to(device)  # [1,1, H,W]  # This is essential in masking and loss computation. Care must be taken for the shape according to the model design.
    
    # %%
    zarr_store = 'data/NYSM.zarr'
    train_val_dates_range = ['2021-01-01T00:00:00', '2021-01-31T23:59:59']
    freq = '5min'
    data_seed = 42

    NYSM_stats = xr.open_dataset('NYSM_variable_stats.nc')
    input_stats = NYSM_stats.sel(variable=[variable])
    target_stats = NYSM_stats.sel(variable=[variable])
    # Standardization
    if transform is not None:
        input_transform = Transform(
            mode=transform,  # 'standard' or 'minmax'
            stats=input_stats,
            feature_axis=None,
        )
        target_transform = Transform(
            mode=transform,  # 'standard' or 'minmax'
            stats=target_stats,
            feature_axis=None,
        )
    else:
        input_transform = None
        target_transform = None

    mode = 'train'
    train_dataset = nowcast_dataset(
        zarr_store,
        variable,
        train_val_dates_range,
        input_sequence_length,
        output_sequence_length,
        freq,
        missing_times=None,
        mode=mode,
        data_seed=data_seed,
        step_size=step_size,
        forecast_offset=forecast_offset
        )

    if is_distributed():
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # shuffle if not using DDP
        pin_memory=True,prefetch_factor=4, persistent_workers=True,
        num_workers=num_workers,
        drop_last=True
    )

    mode = 'val'
    validation_dataset = nowcast_dataset(
        zarr_store,
        variable,
        train_val_dates_range,
        input_sequence_length,
        output_sequence_length,
        freq,
        missing_times=None,
        mode=mode,
        data_seed=data_seed,
        step_size=output_sequence_length,
        forecast_offset=forecast_offset
        )

    if is_distributed():
        validation_sampler = DistributedSampler(validation_dataset)
    else:
        validation_sampler = None
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=validation_sampler,
        shuffle=(validation_sampler is None), # shuffle if not using DDP
        pin_memory=True,prefetch_factor=4, persistent_workers=True,
        num_workers=num_workers,
        drop_last=True
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Data loaded successfully.")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")
    
    # %%
    # === Set up device, model, loss, optimizer ===
    input_resolution = (orography.shape[0], orography.shape[1])
    in_channels = input_sequence_length
    out_channels = output_sequence_length

    if activation_layer == 'gelu':
        act_layer = nn.GELU
    elif activation_layer == 'relu':
        act_layer = nn.ReLU
    elif activation_layer == 'leakyrelu':
        act_layer = nn.LeakyReLU

    if model_name == "DCNN":
        C = 48
        kernel = (7, 7)
        final_kernel = (3, 3)
        n_layers = 7
        model = DCNN(in_channels=in_channels, 
                        out_channels=out_channels, 
                        C=C, 
                        kernel=kernel,
                        final_kernel=final_kernel, 
                        n_layers=n_layers,
                        act_layer=act_layer,
                        hard_enforce_stations=True).to(device)
    elif model_name == "UNet":
        C = 32
        n_layers = 4
        dropout_prob=0.2
        drop_path_prob=0.2
        model = UNet(in_channels=in_channels, 
                        out_channels=out_channels,
                        C=C, 
                        dropout_prob=dropout_prob,
                        drop_path_prob=drop_path_prob,
                        act_layer=act_layer,
                        n_layers=n_layers,
                        hard_enforce_stations=True).to(device)
    
    elif model_name == "SwinT2UNet":
        C = 32
        n_layers = 4
        window_sizes = [8, 8, 4, 4, 2]
        head_dim = 32
        attn_drop = 0.2
        proj_drop = 0.2
        mlp_ratio = 4.0
        model = SwinT2UNet(input_resolution=input_resolution, 
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        C=C, n_layers=n_layers, 
                        window_sizes=window_sizes,
                            head_dim=head_dim,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            mlp_ratio=mlp_ratio,
                            act_layer=act_layer,
                            hard_enforce_stations=True).to(device)
    
    if act_layer == nn.GELU:
            initialize_weights_xavier(model,seed = weights_seed)
    elif act_layer == nn.ReLU:
        initialize_weights_he(model,seed = weights_seed)
    elif act_layer == nn.LeakyReLU:
        initialize_weights_he(model,seed = weights_seed)

    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define the loss criterion and metric here, based on input loss name. The functions are sent to the GPU inside 
    if args.loss_name == "MaskedCharbonnierLoss":
        criterion = MaskedCharbonnierLoss(mask_tensor_expanded,eps=1e-3)
    elif args.loss_name == "MaskedCombinedMAEQuantileLoss":
        criterion = MaskedCombinedMAEQuantileLoss(mask_tensor_expanded, tau=0.95, mae_weight=0.5, quantile_weight=0.5)

    metric = MaskedErrorLoss(mask_tensor_expanded).to(device)

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
                "model_name": model_name,
                "activation_layer": activation_layer,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "weights_seed": weights_seed,
                "loss_name": loss_name,
                "input_sequence_length": input_sequence_length,
                "output_sequence_length": output_sequence_length,
                "train_val_dates_range": train_val_dates_range,
                "transform": transform,
            },
            name=checkpoint_dir[len('checkpoints/'):].replace('/','_'),
            dir="wandb_logs"
        )
    
    # %%
    '''
    # === Checking the data loading bottle neck ===
    print(f"🧪 [Rank {rank}] Testing DataLoader...")
    loader_start = time.time()

    data_bar = tqdm(enumerate(train_dataloader), desc=f"🔄 Rank {rank} DataLoader Test")

    for b_idx, batch in data_bar:
        t0 = time.time()
        input_tensor, target_tensor, in_time, out_time = batch
        input_tensor = input_tensor.to(device, non_blocking=True)
        target_tensor = target_tensor.to(device, non_blocking=True)
        t1 = time.time()

        data_bar.set_postfix({
            "load_time_s": f"{t1 - t0:.4f}",
            "input_shape": str(tuple(input_tensor.shape))
        })

    loader_end = time.time()
    print(f"✅ [Rank {rank}] Completed in {loader_end - loader_start:.2f} seconds.")
    '''
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
        device=device,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        train_sampler=train_sampler, 
        scheduler=scheduler,
        early_stopping=early_stopping,
        mask_tensor_expanded=mask_tensor_expanded,
        input_transform=input_transform,
        target_transform=target_transform,
        resume=resume
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