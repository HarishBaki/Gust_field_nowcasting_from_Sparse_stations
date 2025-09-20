import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

class MaskedErrorLoss(nn.Module):
    """
    Computes masked MSE, RMSE, or MAE.
    Supports:
      - [B, C, H, W]
      - [B, T, H, W, C]
    """
    def __init__(self, mask_tensor_extended):
        super().__init__()
        self.register_buffer("mask_tensor_extended", mask_tensor_extended.float())

    def forward(self, output, target, mode='mae',reduction='mean'):
        """
        mode: 'mse', 'rmse', 'mae'
        reduction: 'mean' or 'none'
        """
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
        
        mask = self.mask_tensor_extended
        if mask.shape != output.shape:
            mask = mask.expand_as(output)

        diff = output - target
        if mode in ['mse', 'rmse']:
            err = diff ** 2
        elif mode == 'mae':
            err = diff.abs()

        masked_err = err * mask
        err_sum = masked_err.reshape(output.shape[0], -1).sum(dim=1)   # [B]
        valid_counts = mask.reshape(output.shape[0], -1).sum(dim=1).clamp(min=1.0)

        per_sample = err_sum / valid_counts

        if mode == 'rmse':
            per_sample = torch.sqrt(per_sample)

        if reduction == 'mean':
            return per_sample.mean()
        elif reduction == 'none':
            return err_sum, valid_counts
        else:
            raise ValueError(f"Unsupported reduction={reduction}")

class MaskedCharbonnierLoss(nn.Module):
    """
    Charbonnier Loss, only over valid (masked) locations.
    Works for inputs of shape [B,C,H,W] or [B,T,H,W,C].
    """
    def __init__(self, mask_tensor_extended, eps=1e-3):
        """
        mask_tensor_extended: torch.Tensor [B, C, H, W] (1=inside NY, 0=outside NY)
        eps: Charbonnier smoothing factor
        """
        super().__init__()
        self.register_buffer("mask_tensor_extended", mask_tensor_extended.float())   # persists on .cuda()/.cpu(), such that the mask_2d devie is used.
        self.eps = eps

    def forward(self, output, target):
        """
        output: [B, 1, H, W] (prediction)
        target: [B, 1, H, W] (target)
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
        
        mask = self.mask_tensor_extended
        if mask.shape != output.shape:
            mask = mask.expand_as(output)

        diff = output - target
        charbonnier = torch.sqrt(diff ** 2 + self.eps ** 2)
        masked_charb = charbonnier * mask
        loss = masked_charb.sum() / mask.sum().clamp(min=1.0)
        return loss

class MaskedPSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) loss, only over valid (masked) locations.
    """
    def __init__(self, mask_2d):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        reduction: elementwise_mean for an overall score, none: for sample wise score.
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())   # persists on .cuda()/.cpu(), such that the mask_2d devie is used.

    def forward(self, x, y,reduction='elementwise_mean'):
        """
        x: [B, 1, H, W] (prediction)
        y: [B, 1, H, W] (target)
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, C, H, W = x.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
        valid_mask = mask.expand(B, C, H, W).float()   # [B, 1, H, W].

        x = x * valid_mask
        y = y * valid_mask
        # Compute min and max only from valid target values
        min_val = y.min()
        max_val = y.max()
        data_range = (min_val.item(), max_val.item())

        psnr = PeakSignalNoiseRatio(reduction=reduction,dim=[1,2,3],data_range=data_range)
        return psnr(x, y)
    
class MaskedSSIM(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) loss, only over valid (masked) locations.
    """
    def __init__(self, mask_2d):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        reduction: elementwise_mean for an overall score, none: for sample wise score.
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())   # persists on .cuda()/.cpu(), such that the mask_2d devie is used.

    def forward(self, x, y,reduction='elementwise_mean'):
        """
        x: [B, 1, H, W] (prediction)
        y: [B, 1, H, W] (target)
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, C, H, W = x.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
        valid_mask = mask.expand(B, C, H, W).float()   # [B, 1, H, W].

        x = x * valid_mask
        y = y * valid_mask
        # Compute min and max only from valid target values
        min_val = y.min()
        max_val = y.max()
        data_range = (min_val.item(), max_val.item())

        ssim = StructuralSimilarityIndexMeasure(reduction=reduction,data_range=data_range)
        return ssim(x, y)

class MaskedCombinedMAEQuantileLoss(nn.Module):
    """
    Combined masked loss: MAE and Quantile Loss (e.g., 95th percentile) for 2D spatial maps
    Applies masking to exclude station locations and non-domain regions.
    """
    def __init__(self, mask_2d, tau=0.95, mae_weight=0.5, quantile_weight=0.5):
        """
        mask_2d: torch.Tensor of shape [H, W], NY domain mask
        tau: quantile level (e.g., 0.95)
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())
        self.tau = tau
        self.mae_weight = mae_weight
        self.quantile_weight = quantile_weight

    def forward(self, output, target, reduction='mean'):
        """
        output: [B, 1, H, W]
        target: [B, 1, H, W]
        station_mask: [B, 1, H, W]
        reduction: 'none', 'mean', or 'global'
        """
        assert reduction in ['mean', 'none', 'global']
        B, C, H, W = output.shape

        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        valid_mask = mask.expand(B, C, H, W).float()   # [B, 1, H, W].

        abs_error = torch.abs(output - target)
        error = target - output
        quantile_error = torch.max(self.tau * error, (self.tau - 1) * error)

        mae_masked = abs_error * valid_mask
        quantile_masked = quantile_error * valid_mask

        mae_sum_per_sample = mae_masked.reshape(B, -1).sum(dim=1)
        quantile_sum_per_sample = quantile_masked.reshape(B, -1).sum(dim=1)
        valid_counts = valid_mask.reshape(B, -1).sum(dim=1).clamp(min=1.0)  # [B]

        mae_per_sample = mae_sum_per_sample / valid_counts
        quantile_per_sample = quantile_sum_per_sample / valid_counts

        combined_per_sample = self.mae_weight * mae_per_sample + self.quantile_weight * quantile_per_sample

        if reduction == 'none':
            return combined_per_sample  # [B]
        elif reduction == 'mean':
            return combined_per_sample.mean()  # scalar
        elif reduction == 'global':
            total_mae = mae_sum_per_sample.sum()
            total_qtl = quantile_sum_per_sample.sum()
            total_count = valid_counts.sum().clamp(min=1.0)
            return (self.mae_weight * total_mae + self.quantile_weight * total_qtl) / total_count  # scalar