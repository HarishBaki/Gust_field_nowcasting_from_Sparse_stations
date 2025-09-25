# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel,dropout_prob=0.2,drop_path_prob=0.0,act_layer=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Dropout2d(p=dropout_prob)
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        out1 = self.block(x)
        out1 = self.drop_path(out1)
        out2 = self.conv11(x)
        out = out1 + out2
        return out
    
class Encoder(nn.Module):
    def __init__(self,in_channels, C, dropout_prob=0.2,drop_path_prob=0.0,act_layer=nn.ReLU, n_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.C = C
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(n_layers):
            if i ==0:
                in_channels = self.in_channels
                out_channels = self.C
            else:
                in_channels = self.C*(2**(i-1))
                out_channels = self.C*(2**i)
            self.blocks.append(ConvBlock(in_channels, out_channels,dropout_prob=dropout_prob,drop_path_prob=drop_path_prob,act_layer=act_layer))
            self.downs.append(nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        skip_connections = []
        for block, down in zip(self.blocks, self.downs):
            x = block(x)
            #print("x convblock shape", x.shape)
            skip_connections.append(x)
            x = down(x)
            #print("x down shape", x.shape)
        return x, skip_connections

class Bottleneck(nn.Module):
    def __init__(self, C,dropout_prob=0.2,drop_path_prob=0.0,act_layer=nn.ReLU, n_layers=4):
        super().__init__()
        self.C = C
        self.n_layers = n_layers
        in_channels = self.C*(2**(n_layers-1))
        out_channels = self.C*(2**n_layers)
        self.block = ConvBlock(in_channels, out_channels,dropout_prob=dropout_prob,drop_path_prob=drop_path_prob,act_layer=act_layer)

    def forward(self, x):
        x = self.block(x)
        #print("x bottleneck shape", x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, C,dropout_prob=0.2,drop_path_prob=0.0,act_layer=nn.ReLU, n_layers=4):
        super().__init__()
        self.out_channels = out_channels
        self.C = C
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(n_layers-1, -1, -1):
            in_channels = self.C*(2**(i+1))
            out_channels = self.C*(2**(i))
            self.ups.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(in_channels, out_channels,dropout_prob=dropout_prob,drop_path_prob=drop_path_prob,act_layer=act_layer))
        self.blocks.append(nn.Conv2d(self.C, self.out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, skip_connections):
        for up, block, skip in zip(self.ups, self.blocks[:-1], skip_connections[::-1]):
            x = up(x)
            #print("x up shape", x.shape)
            x = torch.cat((x, skip), dim=1)
            #print("x cat shape", x.shape)
            x = block(x)
            #print("x block shape", x.shape)
        x = self.blocks[-1](x)
        #print("x final shape", x.shape)
        return x

class UNet(nn.Module):
    '''
    U-Net architecture for spatiotemporal image reconstruction tasks.

    The model is configured entirely from a single `configs` (args) object, 
    which specifies input/output channels, number of layers, activation 
    function, and regularization settings.

    Structure:
    - Encoder: progressively downsamples the input. At each stage, channels 
      are doubled and spatial dimensions are halved.
    - Bottleneck: deepest part of the network, processing compressed 
      representations.
    - Decoder: progressively upsamples. At each stage, features are 
      concatenated with corresponding encoder outputs (skip connections), 
      channels are halved, and spatial dimensions are restored.

    Parameters taken from `configs`:
    - input_sequence_length (int): number of input channels (Tin).
    - output_sequence_length (int): number of output channels (Tout).
    - C (int): base channel dimension used in intermediate layers.
    - n_layers (int): depth of encoder/decoder (number of down/up blocks).
    - dropout_prob (float): dropout probability inside blocks.
    - drop_path_prob (float): stochastic depth (drop path) probability.
    - act_layer (nn.Module): activation function (e.g., nn.ReLU, nn.GELU).
    - hard_enforce_stations (bool, optional): if True, enforces station values 
      at known grid points in the output (application-specific).

    Notes:
    - This implementation follows the general U-shaped design from 
      "Uformer: A General U-Shaped Transformer for Image Restoration" 
      (https://github.com/ZhendongWang6/Uformer).
    - Encoder doubles channels per stage, Decoder mirrors the process 
      with skip connections.
    '''
    def __init__(self, configs):
        super(UNet, self).__init__()
        input_sequence_length = configs.input_sequence_length
        C = configs.C
        output_sequence_length = configs.output_sequence_length
        dropout_prob = configs.dropout_prob
        drop_path_prob = configs.drop_path_prob
        act_layer = configs.act_layer
        n_layers = configs.n_layers
        self.encoder = Encoder(input_sequence_length, C,dropout_prob,drop_path_prob,act_layer, n_layers)
        self.bottleneck = Bottleneck(C, dropout_prob,drop_path_prob,act_layer, n_layers)
        self.decoder = Decoder(output_sequence_length, C, dropout_prob,drop_path_prob,act_layer, n_layers)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return x
    
# %%    
if __name__ == "__main__":
    from types import SimpleNamespace
    from util import initialize_weights_xavier,initialize_weights_he
    # === Define arguments in one place ===
    args = SimpleNamespace(
        input_sequence_length=3,      # instead of in_channels
        output_sequence_length=1,     # instead of out_channels
        C=32,
        dropout_prob=0.2,
        drop_path_prob=0.2,
        act_layer=nn.ReLU,
        n_layers=4,
        hard_enforce_stations=True,
        weights_seed=42
    )

    # === Build model from args ===
    model = UNet(args)

    # === Initialize weights ===
    if args.act_layer == nn.GELU:
        initialize_weights_xavier(model, seed=args.weights_seed)
    elif args.act_layer == nn.ReLU:
        initialize_weights_he(model, seed=args.weights_seed)

    print(model)

    # print the total number of parameters in the model
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    x = torch.randn(1, 3, 256, 288)  # Example input
    output = model(x)
    print("Output shape:", output.shape)  # Should be (1, 1, 256, 2)
    '''
    Already examined the model architecture, intermediate outputs shape and the final output shape.
    ''' 
    # Print weights of the first conv layer
    first_conv_layer = model.encoder.blocks[0].block[0]  # First nn.Conv2d inside the first block
    print("Weights of the first Conv2d layer:")
    print(first_conv_layer.weight.data)
# %%
