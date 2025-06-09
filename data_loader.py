# %%
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time
import os
# %%
class Transform:
    def __init__(self, mode, stats):
        """
        mode: 'standard' or 'minmax'
        stats: 
            if mode == 'standard': {'mean': [...], 'std': [...]}
            if mode == 'minmax': {'min': [...], 'max': [...]}
        """
        self.mode = mode
        if mode == "standard":
            self.param1 = torch.tensor(stats['mean'].values, dtype=torch.float32)  # mean
            self.param2 = torch.tensor(stats['std'].values, dtype=torch.float32)   # std
        elif mode == "minmax":
            self.param1 = torch.tensor(stats['min'].values, dtype=torch.float32)   # min
            self.param2 = torch.tensor(stats['max'].values, dtype=torch.float32)   # max
        else:
            raise ValueError("mode must be 'standard' or 'minmax'")

    def _reshape_params(self, x):
        """
        Reshapes the parameters for broadcasting, based on input dimensions.
        """
        stats_shape = [1, -1] + [1] * (x.ndim - 2)
        p1 = self.param1.reshape(stats_shape).to(x.device)
        p2 = self.param2.reshape(stats_shape).to(x.device)
        return p1, p2

    def __call__(self, x):
        p1, p2 = self._reshape_params(x)
        if self.mode == "standard":
            return (x - p1) / (p2 + 1e-8)
        elif self.mode == "minmax":
            return (x - p1) / (p2 - p1 + 1e-8)

    def inverse(self, x):
        p1, p2 = self._reshape_params(x)
        if self.mode == "standard":
            return x * p2 + p1
        elif self.mode == "minmax":
            return x * (p2 - p1) + p1
    
class nowcast_dataset(Dataset):
    def __init__(self,zarr_store, variable, dates_range, input_window_size, output_window_size, freq,
                 missing_times=None, mode='train',data_seed=42,
                 step_size=1, forecast_offset=0):
        # create a pandas timetime index for the entire training and validation period
        # This will be used to create the input and output samples
        # Unlike the Sparse_to_Dense model, we cannot eliminate the missint instances directly, since we will be dealing with forecasting.
        # So, we need to identify the missing instances in samples, and remove the samples. 
        # The step_size is used to create the time index with non-overlapping time instances.
        reference_dates = pd.date_range(start=dates_range[0], end=dates_range[1], freq=freq)

        # create input and output samples by sliding the input window over the entire training and validation period
        in_samples = []
        out_samples = []
        max_idx  = len(reference_dates) - input_window_size - output_window_size - forecast_offset + 1
        for i in range(0, max_idx, step_size):
            in_samples.append(reference_dates[i:i+input_window_size])
            out_start = i + input_window_size + forecast_offset
            out_end = out_start + output_window_size
            if out_end > len(reference_dates):
                break
            out_samples.append(reference_dates[out_start:out_end])
        in_samples = np.array(in_samples)
        out_samples = np.array(out_samples)
        
        ds = xr.open_zarr(zarr_store)[variable]
        ds = ds.sel(time=slice(*dates_range))

        #original_times = pd.to_datetime(ds.time.values)
        #reference_dates = pd.to_datetime(reference_dates)
        #missing_times = reference_dates.difference(original_times)

        # Sometimes, the original ds will have all the time instances, but an entire time instance could be nans. 
        # Those instances should be indentified externally and pass here as missing_times. 

        if missing_times is not None:
            # Filter out in_samples and out_samples that overlap with missing times
            filtered_in_samples = []
            filtered_out_samples = []
            for in_sample, out_sample in zip(in_samples, out_samples):
                # Check if any time in the input or output window is in the missing times
                if any(time in missing_times for time in in_sample) or any(time in missing_times for time in out_sample):
                    continue  # Skip this sample if it contains a missing time
                filtered_in_samples.append(in_sample)
                filtered_out_samples.append(out_sample)

            # Convert filtered samples to numpy arrays
            filtered_in_samples = np.array(filtered_in_samples)
            filtered_out_samples = np.array(filtered_out_samples)
            #print(filtered_in_samples.shape, filtered_out_samples.shape)
        else:
            # If no missing times, use the original samples
            filtered_in_samples = in_samples
            filtered_out_samples = out_samples

        if mode != 'test':
            rng_data = np.random.default_rng(seed=data_seed)
            years = pd.DatetimeIndex(filtered_in_samples[:, 0]).year
            months = pd.DatetimeIndex(filtered_in_samples[:, 0]).month
            validation_samples = np.zeros(len(filtered_in_samples), dtype=bool)
            for year in np.unique(years):
                for month in range(1, 13):
                    # check if you have enough data in the month
                    month_indices = np.where((years == year) & (months == month))[0]
                    '''
                    One problem we have to deal with is not enough data points in a month.
                    Not always, the number of samples in a month is enough to take 6 days of data for validation.
                    Thus, lets consider 20% of the month data for validation.
                    '''
                    validation_window = int(0.2*len(month_indices))
                    if len(month_indices) == 0:
                        continue
                    try:
                        start_index = rng_data.choice(len(month_indices) - validation_window - 1)
                        #print('start_index:',start_index)
                        validation_indices = month_indices[start_index:start_index + validation_window]
                        validation_samples[validation_indices] = True
                    except:
                        pass
            
            if mode == 'train':
                # Remove validation samples from training samples
                filtered_in_samples = filtered_in_samples[~validation_samples]
                filtered_out_samples = filtered_out_samples[~validation_samples]
            elif mode == 'val':
                # Keep only validation samples
                filtered_in_samples = filtered_in_samples[validation_samples]
                filtered_out_samples = filtered_out_samples[validation_samples]
        else:
            # Keep all samples
            filtered_in_samples = filtered_in_samples
            filtered_out_samples = filtered_out_samples
    
        self.filtered_in_samples = filtered_in_samples
        self.filtered_out_samples = filtered_out_samples
        self.ds = ds

    def __len__(self):
        return len(self.filtered_in_samples)
    
    def __getitem__(self, idx):
        # Get the input and output samples
        in_sample = self.filtered_in_samples[idx]
        out_sample = self.filtered_out_samples[idx]

        # Get the input and output data
        input_tensor = self.ds.sel(time=in_sample).values
        target_tensor = self.ds.sel(time=out_sample).values
    
        # Convert to torch tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)  # [C, H, W]
        target_tensor = torch.tensor(target_tensor, dtype=torch.float32)    # [C, H, W]

        return input_tensor, target_tensor, str(in_sample[0]), str(out_sample[0])

# %%
if __name__ == "__main__":
    # %%
    zarr_store = 'data/NYSM.zarr'
    variable = 'i10fg'
    dates_range = ['2019-01-01T00:00:00', '2019-12-31T23:59:59']
    freq = '5min'
    input_window_size = 3  # 3 hours at every 5 minutes
    output_window_size = 1  # 1 hour at every 5 minutes
    data_seed = 42
    mode = 'train'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # %%
    # Checking data loader with transform
    mask = xr.open_dataset('mask_2d.nc').mask
    mask_tensor = torch.tensor(mask.values).to(device)
    NYSM_stats = xr.open_dataset('NYSM_variable_stats.nc')
    input_stats = NYSM_stats.sel(variable=[variable])
    target_stats = NYSM_stats.sel(variable=[variable])
    # Standardization
    input_transform = Transform(
        mode="minmax",  # 'standard' or 'minmax'
        stats=input_stats
    )
    target_transform = Transform(
        mode="minmax",  # 'standard' or 'minmax'
        stats=target_stats
    )

    # %%
    # Checking data loader without transform
    dataset = nowcast_dataset(
        zarr_store,
        'i10fg',
        dates_range,
        input_window_size,
        output_window_size,
        freq,
        missing_times=None,
        mode=mode,
        data_seed=data_seed,
        step_size=2,
        forecast_offset=1
        )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,num_workers=2, pin_memory=True)
    # %%
    iterator = iter(dataloader)
    # Example usage
    for b in range(3):
        start_time = time.time()
        batch = next(iterator, None)

        if batch is not None:
            input_tensor, target_tensor, input_time_instances, target_time_instances = batch
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            print(f"\n Batch {b+1}")
            print("Input tensor shape:", input_tensor.shape)
            print("Target tensor shape:", target_tensor.shape)
            print("Input time instances:", input_time_instances)
            print("Target time instances:", target_time_instances)

            print('Input and target tensors without transform and before applying mask:')
            for i in range(input_tensor.shape[1]):
                print(f"Input Channel {i} ➜ max: {input_tensor[0, i].max().item():.4f}, min: {input_tensor[0, i].min().item():.4f}")
            for i in range(target_tensor.shape[1]):
                print(f"Target Channel {i} ➜ max: {target_tensor[0, i].max().item():.4f}, min: {target_tensor[0, i].min().item():.4f}")
            
            # Transform input and target
            input_tensor = input_transform(input_tensor)
            target_tensor = target_transform(target_tensor)

            # Apply mask
            input_tensor = torch.where(mask_tensor, input_tensor, 0)
            target_tensor = torch.where(mask_tensor, target_tensor, 0)

            print('Input and target tensors after transform and mask:')
            for i in range(input_tensor.shape[1]):
                print(f"Input Channel {i} ➜ max: {input_tensor[0, i].max().item():.4f}, min: {input_tensor[0, i].min().item():.4f}")
            for i in range(target_tensor.shape[1]):
                print(f"Target Channel {i} ➜ max: {target_tensor[0, i].max().item():.4f}, min: {target_tensor[0, i].min().item():.4f}")
            
            # Inverse transform for input and target
            input_tensor = input_transform.inverse(input_tensor)
            target_tensor = target_transform.inverse(target_tensor)

            # Apply mask
            input_tensor = torch.where(mask_tensor, input_tensor, 0)
            target_tensor = torch.where(mask_tensor, target_tensor, 0)

            print("Inverse transformed tensors shapes:", input_tensor.shape, target_tensor.shape)
            print('Inverse transformed and masked input and target tensors:')
            for i in range(input_tensor.shape[1]):
                print(f"Inverse Input Channel 0 ➜ max: {input_tensor[0][i].max().item():.4f}, min: {input_tensor[0][i].min().item():.4f}")
            for i in range(target_tensor.shape[1]):
                print(f"Inverse Target Channel 0 ➜ max: {target_tensor[0][i].max().item():.4f}, min: {target_tensor[0][i].min().item():.4f}")

        else:
            print(f"Batch {b+1}: No data in this batch.")
        
        end_time = time.time()
        print(f" DataLoader iteration time: {end_time - start_time:.2f} seconds")

# %%
