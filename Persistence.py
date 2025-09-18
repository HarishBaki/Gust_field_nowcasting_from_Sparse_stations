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

import matplotlib.pyplot as plt

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

from losses import MaskedErrorLoss, MaskedTVLoss, MaskedCharbonnierLoss, MaskedCombinedMAEQuantileLoss

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store

"""
# Computing persistance error
- My nowcast_dataset is designed to output input_tensor with a size of [B,input_sequence_length,H,W] and target_tensor with a size of [B,output_sequence_length,H,W]
- For persistance error computation, we set the last instance of inputs as the target for next ouput_window_size. 
- That means, get samples with input_sequence_length = 1, while varying the output_sequence_length over {1:72} for {5min:5hrs}.
- Now, the target_tensor is infact the reference, while input_tensor[:,-1].unsqueeze(1).repeat(1, args.output_sequence_length, 1, 1)  # [B, output_sequence_length, H, W] becomes the persistance model output. 
- Compute the loss for varying output_sequence_lengths 
"""

# %%
if __name__ == "__main__":   
    # %%   
    # This is the main entry point of the script. 
    # === Args for interactive debugging ===
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__') or 'ipykernel' in sys.argv[0]

    # If run interactively, inject some sample arguments
    if is_interactive() or len(sys.argv) == 1:
        sys.argv = [
            "",  # Script name placeholder
            "--prediction_dir", "Predictions",
            "--data_type","NYSM",
            "--zarr_store", "data/NYSM.zarr",
            "--start_date", "2023-01-01T00:00:00",
            "--end_date", "2023-01-31T23:59:59",
            "--freq", "5min",
            "--output_sequence_length", "72",
            "--batch_size", "32",
            "--num_workers", "16",
        ]
        print("DEBUG: Using injected args:", sys.argv)
    
    # === Argument parsing ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", type=str, required=True,
                        help="Directory to save predictions and results.")
    parser.add_argument("--data_type", type=str, default="NYSM",
                        help="Type of data (e.g., NYSM).")
    parser.add_argument("--zarr_store", type=str, required=True,
                        help="Path to the Zarr data store.")
    parser.add_argument("--start_date", type=str, required=True,
                        help="Start date for testing (e.g., '2023-01-01T00:00:00').")
    parser.add_argument("--end_date", type=str, required=True,
                        help="End date for testing (e.g., '2023-01-31T23:59:59').")
    parser.add_argument("--freq", type=str, default="5min",
                        help="Data frequency (e.g., '5min').")
    parser.add_argument("--output_sequence_length", type=int, default=72,
                        help="Number of time steps to predict.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for DataLoader.")
    args = parser.parse_args()

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

    args.device = device
    # %%
    # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc')
    RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
    RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
    orography = orography.set_coords(['latitude', 'longitude'])
    orography = orography.orog

    mask = xr.open_dataset('mask_2d.nc').mask
    mask_tensor = torch.tensor(mask.values.astype(np.float32),device=args.device)  # [H, W], 1=valid, 0=invalid

    # Load NYSM station data
    nysm = pd.read_csv('nysm.csv')
    # NYSM station lat/lon
    nysm_latlon = np.stack([
        nysm['lat [degrees]'].values,
        (nysm['lon [degrees]'].values + 360) % 360
    ], axis=-1) # shape: (N, 2)

    exclude_indices = [65, 102] # Exclude these indices, since they are falling outside the NYS mask region. 

    # Precompute grid KDTree
    grid_points = np.stack([RTMA_lat.ravel(), RTMA_lon.ravel()], axis=-1)
    tree = cKDTree(grid_points)
    # Query the station locations
    _, indices_flat = tree.query(nysm_latlon)
    # Convert flat indices to 2D (y, x)
    y_indices, x_indices = np.unravel_index(indices_flat, RTMA_lat.shape)

    # %%
    test_dates_range = [args.start_date, args.end_date]
    test_dataset = nowcast_dataset(
        args.zarr_store,
        'i10fg',
        test_dates_range,
        1,
        args.output_sequence_length,
        args.freq,
        missing_times=None,
        mode='test',
        step_size=1,
        forecast_offset=0
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    print("Test dataset length:", len(test_dataset))

    metric = MaskedErrorLoss(mask_tensor).to(args.device)
    total_ae, total_count_ae = 0.0, 0.0
    total_se, total_count_se = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="[Test]"):
            inputs, targets, _, _ = batch
            inputs = inputs.unsqueeze(-1).to(args.device)   # (B, T_in, H, W, C)
            targets = targets.unsqueeze(-1).to(args.device) # (B, T_out, H, W, C)

            # Persistence prediction: repeat last input
            persistence_pred = inputs[:, -1].unsqueeze(1).repeat(
                1, args.output_sequence_length, 1, 1, 1
            )

            # MAE
            err_sum, counts = metric(persistence_pred, targets, mode='mae', reduction='none')
            total_ae += err_sum.sum().item()
            total_count_ae += counts.sum().item()

            # MSE
            err_sum, counts = metric(persistence_pred, targets, mode='mse', reduction='none')
            total_se += err_sum.sum().item()
            total_count_se += counts.sum().item()

        mae = total_ae / total_count_ae
        mse = total_se / total_count_se
        rmse = np.sqrt(mse)

    # save to a text file
    target_path = f'{args.prediction_dir}/Persistence/{args.data_type}/freq_{args.freq}'
    os.makedirs(target_path, exist_ok=True)
    with open(f'{target_path}/Horizon_{args.output_sequence_length}.txt', 'w') as f:
        f.write(f'MAE: {mae:.4f}\n')
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        
    # %%
    if dist.is_initialized():
        dist.destroy_process_group()