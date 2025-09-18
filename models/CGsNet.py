# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser

# ------------------------------
# Core Model (Pure PyTorch)
# ------------------------------
class CGsNet(nn.Module):
    def __init__(
        self,
        height,
        width,
        input_length,
        target_length,
        downscale_factor=4,
        num_channels_in=1,
        num_channels_out=1,
        cnn_hidden_size=64,
        rnn_input_dim=64,
        phycell_hidden_dims=[64],
        kernel_size_phycell=3,
        convlstm_hidden_dims=[64],
        kernel_size_convlstm=3,
    ):
        super().__init__()

        # Store only what we actually use inside the modules/forward
        self.input_length = input_length
        self.target_length = target_length
        self.convlstm_hidden_dims = convlstm_hidden_dims
        self.phycell_hidden_dims = phycell_hidden_dims
        self.rnn_input_dim = rnn_input_dim

        assert height % downscale_factor == 0, "downscale_height should divide height"
        assert width % downscale_factor == 0, "downscale_width should divide width"

        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.rnn_cell_height = height // downscale_factor
        self.rnn_cell_width = width // downscale_factor

        self.encoder = EncoderRNN(
            self.num_channels_in,
            cnn_hidden_size,
            self.num_channels_out,
            self.rnn_cell_height,
            self.rnn_cell_width,
            rnn_input_dim,
            phycell_hidden_dims,
            kernel_size_phycell,
            convlstm_hidden_dims,
            kernel_size_convlstm,
            downscale_factor,
        )
        self.decoder = DecoderRNN_ATT(
            self.num_channels_in,
            cnn_hidden_size,
            self.num_channels_out,
            self.rnn_cell_height,
            self.rnn_cell_width,
            rnn_input_dim,
            phycell_hidden_dims,
            kernel_size_phycell,
            convlstm_hidden_dims,
            kernel_size_convlstm,
            downscale_factor,
            input_length=input_length,
        )

        self.layers_phys = len(phycell_hidden_dims)
        self.layers_convlstm = len(convlstm_hidden_dims)

    def forward(self, input_tensor, target_tensor=None, use_teacher_forcing=False):
        """
        input_tensor: (B, Tin, C, H, W)
        target_tensor: (B, Tout-1, C_out, H, W) if teacher forcing
        returns:
            encoder_frames (B, Tin-1, C_out?, H, W)  [as in original]
            decoder_frames (B, Tout,  C_out,  H, W)
        """
        device = input_tensor.device
        batch = input_tensor.shape[0]

        encoder_frames = []
        decoder_frames = []
        encoder_att = []

        # Init hidden states
        h_t, c_t, phys_h_t = [], [], []
        for i in range(self.layers_convlstm):
            zeros = torch.zeros(
                [batch, self.convlstm_hidden_dims[i], self.rnn_cell_height, self.rnn_cell_width],
                device=device,
            )
            h_t.append(zeros)
            c_t.append(zeros.clone())
        for _ in range(self.layers_phys):
            phys_h_t.append(
                torch.zeros([batch, self.rnn_input_dim, self.rnn_cell_height, self.rnn_cell_width], device=device)
            )

        # Encoder pass over all but last input frame
        for ei in range(self.input_length - 1):
            h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image, output_att = self.encoder(
                input_tensor[:, ei],
                first_timestep=(ei == 0),
                h_t=h_t,
                c_t=c_t,
                phys_h_t=phys_h_t,
            )
            encoder_att.append(output_att)
            encoder_frames.append(output_image)

        # Last encoder step (no first_timestep flag)
        h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image, output_att = self.encoder(
            input_tensor[:, -1],
            first_timestep=False,
            h_t=h_t,
            c_t=c_t,
            phys_h_t=phys_h_t,
        )
        encoder_att.append(output_att)
        encoder_att = torch.stack(encoder_att, dim=1)  # (B, Tin, C_att, H, W)
        decoder_frames.append(output_image[:, : self.num_channels_out])

        # Decoder steps
        for di in range(self.target_length - 1):
            if use_teacher_forcing and target_tensor is not None:
                decoder_input = target_tensor[:, di]
            else:
                decoder_input = output_image

            h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image = self.decoder(
                decoder_input,
                encoder_att,
                first_timestep=(di == 0),
                h_t=h_t,
                c_t=c_t,
                phys_h_t=phys_h_t,
            )
            decoder_frames.append(output_image)

        encoder_frames = torch.stack(encoder_frames, dim=1) if len(encoder_frames) > 0 else None
        decoder_frames = torch.stack(decoder_frames, dim=1)

        return encoder_frames, decoder_frames


# ------------------------------
# Submodules (Pure PyTorch)
# ------------------------------
class EncoderRNN(nn.Module):
    def __init__(
        self,
        num_channels_in,
        cnn_hidden_size,
        num_channels_out,
        rnn_cell_height,
        rnn_cell_width,
        rnn_input_dim,
        phycell_hidden_dims,
        kernel_size_phycell,
        convlstm_hidden_dims,
        kernel_size_convlstm,
        downscale=4,
    ):
        super().__init__()
        if downscale == 4:
            self.encoder_E = encoder_4E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_4D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 16:
            self.encoder_E = encoder_16E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_16D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 30:
            self.encoder_E = encoder_30E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_30D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 32:
            self.encoder_E = encoder_32E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_32D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        else:
            raise ValueError("downscale must be one of [4, 16, 30, 32]")

        self.decoder_att = nn.Conv2d(
            in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=1, stride=1, padding=0
        )

        self.encoder_Ep = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.encoder_Er = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.decoder_Dp = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.decoder_Dr = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)

        self.phycell = PhyCell(
            input_shape=(rnn_cell_height, rnn_cell_width),
            input_dim=rnn_input_dim,
            F_hidden_dims=phycell_hidden_dims,
            kernel_size=(kernel_size_phycell, kernel_size_phycell),
        )
        self.convcell = ConvLSTM(
            input_shape=(rnn_cell_height, rnn_cell_width),
            input_dim=rnn_input_dim,
            hidden_dims=convlstm_hidden_dims,
            kernel_size=(kernel_size_convlstm, kernel_size_convlstm),
        )

    def forward(self, input_, first_timestep=False, h_t=None, c_t=None, phys_h_t=None):
        input_ = self.encoder_E(input_)
        input_phys = self.encoder_Ep(input_)
        input_conv = self.encoder_Er(input_)

        phys_h_t, output1 = self.phycell(input_phys, first_timestep, phys_h_t)
        (h_t, c_t), output2 = self.convcell(input_conv, first_timestep, h_t, c_t)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr
        output_image = torch.sigmoid(self.decoder_D(concat))
        att_image = self.decoder_att(concat)
        return h_t, c_t, phys_h_t, out_phys, out_conv, output_image, att_image


class DecoderRNN_ATT(nn.Module):
    def __init__(
        self,
        num_channels_in,
        cnn_hidden_size,
        num_channels_out,
        rnn_cell_height,
        rnn_cell_width,
        rnn_input_dim,
        phycell_hidden_dims,
        kernel_size_phycell,
        convlstm_hidden_dims,
        kernel_size_convlstm,
        downscale=4,
        input_length=6,
    ):
        super().__init__()
        if downscale == 4:
            self.encoder_E = encoder_4E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_4D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 16:
            self.encoder_E = encoder_16E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_16D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 30:
            self.encoder_E = encoder_30E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_30D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 32:
            self.encoder_E = encoder_32E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_32D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        else:
            raise ValueError("downscale must be one of [4, 16, 30, 32]")

        self.encoder_Ep = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.encoder_Er = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.decoder_Dp = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.decoder_Dr = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)

        # Attention over encoder sequence
        self.att = nn.Conv2d(in_channels=rnn_input_dim, out_channels=input_length, kernel_size=1, stride=1, padding=0)
        self.att_combine = nn.Conv2d(
            in_channels=2 * rnn_input_dim, out_channels=rnn_input_dim, kernel_size=1, stride=1, padding=0
        )

        self.phycell = PhyCell(
            input_shape=(rnn_cell_height, rnn_cell_width),
            input_dim=rnn_input_dim,
            F_hidden_dims=phycell_hidden_dims,
            kernel_size=(kernel_size_phycell, kernel_size_phycell),
        )
        self.convcell = ConvLSTM(
            input_shape=(rnn_cell_height, rnn_cell_width),
            input_dim=rnn_input_dim,
            hidden_dims=convlstm_hidden_dims,
            kernel_size=(kernel_size_convlstm, kernel_size_convlstm),
        )

    def forward(self, input_, output_seqs, first_timestep=False, h_t=None, c_t=None, phys_h_t=None):
        input_ = self.encoder_E(input_)
        att_weight = F.softmax(self.att(input_), dim=1)
        # output_seqs expected shape: (B, T, C, H, W)
        att_applied = torch.sum(att_weight.unsqueeze(2) * output_seqs, dim=1)
        input_ = self.att_combine(torch.cat([input_, att_applied], dim=1))

        input_phys = self.encoder_Ep(input_)
        input_conv = self.encoder_Er(input_)

        phys_h_t, output1 = self.phycell(input_phys, first_timestep, phys_h_t)
        (h_t, c_t), output2 = self.convcell(input_conv, first_timestep, h_t, c_t)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr
        output_image = self.decoder_D(concat)
        return h_t, c_t, phys_h_t, out_phys, out_conv, output_image


class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super().__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.GroupNorm(4, F_hidden_dim),
            nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=1, padding=0),
        )

        self.convgate = nn.Conv2d(
            in_channels=self.input_dim + self.input_dim,
            out_channels=self.input_dim,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=self.bias,
        )

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)
        next_hidden = hidden_tilde + K * (x - hidden_tilde)
        return next_hidden


class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, kernel_size):
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = len(F_hidden_dims)
        self.kernel_size = kernel_size

        cell_list = []
        for i in range(self.n_layers):
            cell_list.append(
                PhyCell_Cell(input_dim=input_dim, F_hidden_dim=self.F_hidden_dims[i], kernel_size=self.kernel_size)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False, h_t=None):
        # h_t is list of hidden tensors per layer
        if first_timestep and h_t is not None:
            H = h_t
        else:
            # Expect caller to provide h_t except for very first timestep (handled upstream)
            H = h_t

        for j, cell in enumerate(self.cell_list):
            if j == 0:
                H[j] = cell(input_, H[j])
            else:
                H[j] = cell(H[j - 1], H[j])
        return H, H


class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        super().__init__()
        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, kernel_size):
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        self.kernel_size = kernel_size

        cell_list = []
        for i in range(self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                ConvLSTM_Cell(
                    input_shape=self.input_shape,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_size,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False, h_t=None, c_t=None):
        if first_timestep and (h_t is not None) and (c_t is not None):
            H = h_t
            C = c_t
        else:
            H = h_t
            C = c_t

        for j, cell in enumerate(self.cell_list):
            if j == 0:
                H[j], C[j] = cell(input_, (H[j], C[j]))
            else:
                H[j], C[j] = cell(H[j - 1], (H[j], C[j]))

        return (H, C), H


# ------------------------------
# Building Blocks (Encoders/Decoders)
# ------------------------------
class dcgan_conv(nn.Module):
    def __init__(self, channels_in, channels_out, stride, kernel_size=3, padding=1):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(16, channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.down_conv(input_)


class dcgan_upconv(nn.Module):
    def __init__(self, channels_in, channels_out, stride, kernel_size=3, padding=1, output_padding=0):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.GroupNorm(16, channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.up_conv(input_)


class encoder_4E(nn.Module):
    def __init__(self, nchannels_in=1, nchannels_out=64):
        super().__init__()
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, stride=2)
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=1)
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out, stride=2)

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class encoder_16E(nn.Module):
    def __init__(self, nchannels_in=1, nchannels_out=128):
        super().__init__()
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, stride=2)
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=2)
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=2)
        self.c4 = dcgan_conv(nchannels_out // 2, nchannels_out, stride=2)

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        return h4


class decoder_16D(nn.Module):
    def __init__(self, nchannels_in=128, nchannels_out=1):
        super().__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, stride=2, output_padding=1)
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=2, output_padding=1)
        self.upc3 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=2, output_padding=1)
        self.upc4 = nn.ConvTranspose2d(
            in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        return d4


class encoder_30E(nn.Module):
    def __init__(self, nchannels_in=1, nchannels_out=64):
        super().__init__()
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, kernel_size=7, stride=5, padding=1)
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, kernel_size=5, stride=3, padding=1)
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_4D(nn.Module):
    def __init__(self, nchannels_in=64, nchannels_out=1):
        super().__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, stride=2, output_padding=1)
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=1)
        self.upc3 = nn.ConvTranspose2d(
            in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class decoder_30D(nn.Module):
    def __init__(self, nchannels_in=64, nchannels_out=1):
        super().__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, kernel_size=4, stride=2, padding=1)
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, kernel_size=5, stride=3, padding=1)
        self.upc3 = nn.ConvTranspose2d(
            in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=7, stride=5, padding=1
        )

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_32E(nn.Module):
    def __init__(self, nchannels_in=1, nchannels_out=128):
        super().__init__()
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, kernel_size=5, stride=4, padding=1)
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, kernel_size=3, stride=2, padding=1)
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out, kernel_size=5, stride=4, padding=1)

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_32D(nn.Module):
    def __init__(self, nchannels_in=128, nchannels_out=1):
        super().__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, kernel_size=6, stride=4, padding=1)
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, kernel_size=4, stride=2, padding=1)
        self.upc3 = nn.ConvTranspose2d(
            in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=6, stride=4, padding=1
        )

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):
    def __init__(self, nchannels_in=64, nchannels_out=64):
        super().__init__()
        self.c1 = dcgan_conv(nchannels_in, nchannels_out, stride=1)
        self.c2 = dcgan_conv(nchannels_out, nchannels_out, stride=1)

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):
    def __init__(self, nchannels_in=64, nchannels_out=64):
        super().__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in, stride=1)
        self.upc2 = dcgan_upconv(nchannels_in, nchannels_out, stride=1)

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        return d2

# %%
if __name__ == "__main__":
    # %%
    height=64
    width=64
    input_length=36
    target_length=36
    downscale_factor=4
    num_channels_in=1
    num_channels_out=1
    cnn_hidden_size=64
    rnn_input_dim=64
    phycell_hidden_dims=[64]
    kernel_size_phycell=3
    convlstm_hidden_dims=[64]
    kernel_size_convlstm=3
    # %%
    # Instantiate model
    model = CGsNet(
        height=height,
        width=width,
        input_length=input_length,
        target_length=target_length,
        downscale_factor=downscale_factor,
        num_channels_in=num_channels_in,
        num_channels_out=num_channels_out,
        cnn_hidden_size=cnn_hidden_size,
        rnn_input_dim=rnn_input_dim,
        phycell_hidden_dims=phycell_hidden_dims,
        kernel_size_phycell=kernel_size_phycell,
        convlstm_hidden_dims=convlstm_hidden_dims,
        kernel_size_convlstm=kernel_size_convlstm,
    )

    # %%
    # Quick sanity run
    B, Tin = 2, input_length
    C_in = num_channels_in
    H, W = height, width

    x = torch.randn(B, Tin, C_in, H, W)
    enc_frames, dec_frames = model(x, None, False)

    print(
        "encoder_frames:", None if enc_frames is None else tuple(enc_frames.shape),
        "\ndecoder_frames:", tuple(dec_frames.shape),
    )
# %%
