# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class DCNN(nn.Module):
    '''
    Deep Convolutional Neural Network (DCNN) for field reconstruction from sparse sensors.

    Config-driven version: all hyperparameters are taken from a single `configs` object.

    Structure:
    - Multiple convolutional blocks, each followed by activation.
    - Final convolution maps intermediate features to target output channels.

    Parameters expected in `configs`:
    - input_sequence_length (int): number of input channels (Tin).
    - output_sequence_length (int): number of output channels (Tout).
    - C (int): number of channels in the intermediate layers.
    - kernel (tuple of int): kernel size for intermediate conv layers.
    - final_kernel (tuple of int): kernel size for final conv layer.
    - n_layers (int): number of convolutional layers.
    - act_layer (nn.Module): activation function (e.g., nn.ReLU, nn.GELU).
    '''

    def __init__(self, configs):
        super().__init__()
        self.in_channels = configs.input_sequence_length
        self.out_channels = configs.output_sequence_length
        self.C = configs.C
        self.kernel = configs.kernel
        self.final_kernel = configs.final_kernel
        self.n_layers = configs.n_layers
        self.act_layer = configs.act_layer

        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = self.in_channels if i == 0 else self.C
            out_ch = self.C
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=self.kernel,
                          padding=(self.kernel[0] // 2, self.kernel[1] // 2)),
                self.act_layer()
            ))

        self.final_conv = nn.Conv2d(
            self.C, self.out_channels,
            kernel_size=self.final_kernel, padding=1
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

# %%
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from types import SimpleNamespace
    from util import initialize_weights_xavier, initialize_weights_he

    # === Define configs ===
    configs = SimpleNamespace(
        input_sequence_length=3,
        output_sequence_length=1,
        C=48,
        kernel=(7, 7),
        final_kernel=(3, 3),
        n_layers=7,
        act_layer=nn.GELU,
        weights_seed=42
    )

    # === Build model ===
    model = DCNN(configs)

    # === Weight initialization ===
    if configs.act_layer == nn.GELU:
        initialize_weights_xavier(model, seed=configs.weights_seed)
    elif configs.act_layer == nn.ReLU:
        initialize_weights_he(model, seed=configs.weights_seed)

    print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    # === Test forward ===
    x = torch.randn(1, configs.input_sequence_length, 256, 288)
    output = model(x)
    print("Output shape:", output.shape)

    # Print weights of the first conv layer
    first_conv_layer = model.blocks[0][0]
    print("Weights of the first Conv2d layer:")
    print(first_conv_layer.weight.data)
# %%
