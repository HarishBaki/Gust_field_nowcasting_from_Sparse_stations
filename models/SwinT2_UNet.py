# %%
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.swin_transformer_v2 import SwinTransformerV2Block

def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, height, width]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1).contiguous()


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2).contiguous()

class InputProj(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        # Input projection
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding=self.kernel_size//2),
            act_layer()
        )
        if norm_layer is not None:
            self.norm = norm_layer(self.out_channels)
        else:
            self.norm = None

    def forward(self, x):
        # INput shape is B C H W
        x = self.proj(x)  # B C H W
        x = bchw_to_bhwc(x) # B H W C
        if self.norm is not None:
            x = self.norm(x)
        return x # Output shape is B H W C

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channels=32, out_channels=1, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(self.out_channels)
        else:
            self.norm = None

    def forward(self, x):
        # Input shape is B H W C
        x = bhwc_to_bchw(x) # Convert to B C H W
        x = self.proj(x)    # B C H W
        if self.norm is not None:
            x = self.norm(x)
        return x    # Output shape is B C H W

class DownSample(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=4, stride=2, padding = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.downsample = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        # Input shape is B H W C
        x = bhwc_to_bchw(x) # Convert to B C H W
        x = self.downsample(x)
        x = bchw_to_bhwc(x) # Convert back to B H W C
        return x    # Output shape is B H W C

class UpSample(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # Input shape is B H W C
        x = bhwc_to_bchw(x) # Convert to B C H W
        x = self.upsample(x)
        x = bchw_to_bhwc(x) # Convert back to B H W C
        return x    # Output shape is B H W C

class Encoder(nn.Module):
    def __init__(self, input_resolution = (256,288), C=32, window_sizes = [8,8,4,4], head_dim=32, n_layers=4, 
                 attn_drop=0.2, proj_drop=0.2,mlp_ratio=4.0,act_layer=nn.GELU):

        super().__init__()
        self.input_resolution = input_resolution
        self.C = C
        self.n_layers = n_layers
        self.window_sizes = window_sizes
        self.head_dim = head_dim
        self.swin_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(n_layers):
            input_resolution = (self.input_resolution[0] // (2**(i)), self.input_resolution[1] // (2**(i)))
            dim = self.C*(2**i)
            num_heads = dim // self.head_dim
            input_channels = self.C*(2**i)
            out_channels = self.C*(2**(i+1))
            window_size = self.window_sizes[i]
            #  Swin transformer block
            self.swin_blocks.append(nn.Sequential(
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer),
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size//2,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer)))
            # Downsample block
            self.downs.append(DownSample(in_channels=input_channels, out_channels=out_channels))

    def forward(self, x):
        # Input shape is B H W C
        skip_connections = []
        for swin_block, down in zip(self.swin_blocks, self.downs):
            x = swin_block(x)   # B H W C
            skip_connections.append(x)
            x = down(x) # B H W C
        return x, skip_connections

class Bottleneck(nn.Module):
    def __init__(self, C, input_resolution, n_layers=4,head_dim=32, window_sizes=[8,8,4,4,2], 
                 attn_drop=0.2, proj_drop=0.2,mlp_ratio=4.0,act_layer=nn.GELU):
        super().__init__()
        self.C = C
        self.input_resolution = input_resolution
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.window_sizes = window_sizes
        i = n_layers
        input_resolution = (self.input_resolution[0] // (2**(i)), self.input_resolution[1] // (2**(i)))
        dim = self.C*(2**i)
        num_heads = dim // self.head_dim
        window_size = self.window_sizes[i]
        self.bottleneck =  nn.Sequential(
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer),
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size//2,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer))
        
    def forward(self, x):
        # Input shape is B H W C
        x = self.bottleneck(x)  # B H W C
        return x

class Decoder(nn.Module):
    def __init__(self, input_resolution = (256,288), C=32, window_sizes = [8,8,4,4], head_dim=32, n_layers=4,
                 attn_drop=0.2, proj_drop=0.2,mlp_ratio=4.0,act_layer=nn.GELU):
        super().__init__()
        self.input_resolution = input_resolution
        self.C = C
        self.n_layers = n_layers
        self.window_sizes = window_sizes
        self.head_dim = head_dim
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.swin_blocks = nn.ModuleList()

        for i in range(n_layers-1,-1,-1):
            input_resolution = (self.input_resolution[0] // (2**(i)), self.input_resolution[1] // (2**(i)))
            dim = self.C*(2**i)
            num_heads = dim // self.head_dim
            input_channels = self.C*(2**(i+1))
            out_channels = self.C*(2**(i))
            window_size = self.window_sizes[i]
            # Up blocks
            self.ups.append(UpSample(in_channels=input_channels, out_channels=out_channels))
            # conv blocks
            self.convs.append(nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
            #  Swin transformer block
            self.swin_blocks.append(nn.Sequential(
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer),
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size//2,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer)))

    def forward(self, x,skip_connections):
        # Input shape is B H W C
        for up, conv, swin_block, skip in zip(self.ups, self.convs, self.swin_blocks, skip_connections[::-1]):
            x = up(x)   # B H W C
            x = torch.cat((x, skip), dim=-1).contiguous()    # Concatenate along the channel dimension, that is -c
            x = bchw_to_bhwc(conv(bhwc_to_bchw(x))) # B H W C
            x = swin_block(x)
        return x

class SwinT2UNet(nn.Module):
    '''
    Swin Transformer V2 U-Net for image-to-image reconstruction.

    Config-driven version: all hyperparameters are taken from a single `configs` object.

    Structure:
    - Input projection → Encoder → Bottleneck → Decoder → Output projection.
    - Encoder downsamples (halving resolution per stage, doubling channels).
    - Decoder upsamples (restoring resolution, concatenating skip connections).
    - Bottleneck captures long-range dependencies at lowest resolution.

    Parameters expected in `configs`:
    - input_resolution (tuple of int): (H, W)
    - input_sequence_length (int): input channels (Tin)
    - output_sequence_length (int): output channels (Tout)
    - C (int): base channel dimension
    - n_layers (int): number of encoder/decoder stages
    - window_sizes (list[int]): attention window sizes
    - head_dim (int): dimension per attention head
    - attn_drop (float): attention dropout rate
    - proj_drop (float): projection dropout rate
    - mlp_ratio (float): MLP expansion ratio
    - act_layer (nn.Module): activation (e.g. nn.GELU)
    '''

    def __init__(self, configs):
        super().__init__()
        self.input_resolution = configs.img_size
        self.input_sequence_length = configs.input_sequence_length
        self.output_sequence_length = configs.output_sequence_length
        self.C = configs.C
        self.n_layers = configs.n_layers
        self.window_sizes = configs.window_sizes
        self.head_dim = configs.head_dim

        self.input_proj = InputProj(self.input_sequence_length, self.C, act_layer=configs.act_layer)
        self.encoder = Encoder(
            input_resolution=self.input_resolution, C=self.C,
            window_sizes=self.window_sizes, head_dim=self.head_dim,
            n_layers=self.n_layers, attn_drop=configs.attn_drop,
            proj_drop=configs.proj_drop, mlp_ratio=configs.mlp_ratio,
            act_layer=configs.act_layer
        )
        self.bottleneck = Bottleneck(
            C=self.C, input_resolution=self.input_resolution,
            n_layers=self.n_layers, window_sizes=self.window_sizes,
            head_dim=self.head_dim, attn_drop=configs.attn_drop,
            proj_drop=configs.proj_drop, mlp_ratio=configs.mlp_ratio,
            act_layer=configs.act_layer
        )
        self.decoder = Decoder(
            input_resolution=self.input_resolution, C=self.C,
            window_sizes=self.window_sizes, head_dim=self.head_dim,
            n_layers=self.n_layers, attn_drop=configs.attn_drop,
            proj_drop=configs.proj_drop, mlp_ratio=configs.mlp_ratio,
            act_layer=configs.act_layer
        )
        self.output_proj = OutputProj(self.C, self.output_sequence_length)

    def forward(self, x):
        # Input: [B, C, H, W]

        x = self.input_proj(x)             # [B, H, W, C]
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = self.output_proj(x)            # [B, C, H, W]

        return x

# %%
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from types import SimpleNamespace
    from util import initialize_weights_xavier, initialize_weights_he

    # === Define configs ===
    configs = SimpleNamespace(
        img_size=(256, 288),
        input_sequence_length=3,   # instead of in_channels
        output_sequence_length=1,  # instead of out_channels
        C=32,
        n_layers=4,
        window_sizes=[8, 8, 4, 4, 2],
        head_dim=32,
        attn_drop=0.2,
        proj_drop=0.2,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        weights_seed=42
    )

    # === Build model ===
    model = SwinT2UNet(configs)

    # === Weight initialization ===
    if configs.act_layer == nn.GELU:
        initialize_weights_xavier(model, seed=configs.weights_seed)
    elif configs.act_layer == nn.ReLU:
        initialize_weights_he(model, seed=configs.weights_seed)

    print(model)  # Print architecture
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    # === Test forward ===
    x = torch.randn(1, configs.input_sequence_length,
                    configs.img_size[0], configs.img_size[1])
    print("Input shape:", x.shape)
    output = model(x)
    print("Output shape:", output.shape)

    # === Inspect weights of first conv layer ===
    first_conv_layer = model.input_proj.proj[0]
    print("Weights of the first Conv2d layer:")
    print(first_conv_layer.weight.data)
# %%
