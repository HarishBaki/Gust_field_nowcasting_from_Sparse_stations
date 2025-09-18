import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class CGsNet(pl.LightningModule):

    def __init__(self, height, width, input_length, target_length, downscale_factor, learning_rate, loss_fx,
                 input_class, predict_class, predict_class_vmax, add_video, weights_prec, thresholds_prec,
                 weights_radar, thresholds_radar, visual_prec_vmin, visual_prec_vmax, visual_train_steps,
                 visual_val_steps, train_log_steps, val_log_steps, test_save_path, cnn_hidden_size, rnn_input_dim,
                 phycell_hidden_dims, kernel_size_phycell, convlstm_hidden_dims, kernel_size_convlstm,
                 lr_scheduler_mode,
                 lr_scheduler_patience, lr_scheduler_factor, lr_scheduler_monitor, lr_scheduler_frequency,
                 sampling_changing_rate_epoch, param_a, param_b, param_c, param_d, **kwargs):

        super(CGsNet, self).__init__()
        self.save_hyperparameters()

        assert height % downscale_factor == 0, "downscale_width should be int!"
        assert width % downscale_factor == 0, "downscale_width should be int!"
        self.num_channels_in = len(input_class)
        self.num_channels_out = len(predict_class)
        self.rnn_cell_height = height // downscale_factor
        self.rnn_cell_width = width // downscale_factor

        self.encoder = EncoderRNN(self.num_channels_in, cnn_hidden_size, self.num_channels_out,
                                  self.rnn_cell_height,
                                  self.rnn_cell_width, rnn_input_dim, phycell_hidden_dims, kernel_size_phycell,
                                  convlstm_hidden_dims, kernel_size_convlstm, downscale_factor)
        self.decoder = DecoderRNN_ATT(self.num_channels_in, cnn_hidden_size, self.num_channels_out,
                                      self.rnn_cell_height,
                                      self.rnn_cell_width,rnn_input_dim, phycell_hidden_dims, kernel_size_phycell,
                                      convlstm_hidden_dims, kernel_size_convlstm, downscale_factor,
                                      input_length=input_length)

        self.layers_phys = len(phycell_hidden_dims)
        self.layers_convlstm = len(convlstm_hidden_dims)

    def forward(self, input_tensor, target_tensor, use_teacher_forcing=False, **kwargs):
        batch = input_tensor.shape[0]
        encoder_frames = []
        decoder_frames = []
        encoder_att = []
        h_t = []
        c_t = []
        phys_h_t = []

        for i in range(self.layers_convlstm):
            zeros = torch.zeros(
                [batch, self.hparams.convlstm_hidden_dims[i], self.rnn_cell_height, self.rnn_cell_width],
                device=self.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.layers_phys):
            phys_h_t.append(torch.zeros([batch, self.hparams.rnn_input_dim, self.rnn_cell_height, self.rnn_cell_width],
                                        device=self.device))

        for ei in range(self.hparams.input_length - 1):
            h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image, output_att = self.encoder(input_tensor[:, ei], ei == 0,
                                                                                        h_t, c_t, phys_h_t)
            encoder_att.append(output_att)
            encoder_frames.append(output_image)

        h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image, output_att = self.encoder(input_tensor[:, -1])
        encoder_att.append(output_att)
        encoder_att = torch.stack(encoder_att, dim=1)
        decoder_frames.append(output_image[:, :self.num_channels_out])

        for di in range(self.hparams.target_length - 1):
            if use_teacher_forcing:
                decoder_input = target_tensor[:, di]
            else:
                decoder_input = output_image
            h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image = self.decoder(decoder_input, encoder_att,
                                                                                        di == 0, h_t,
                                                                                        c_t,
                                                                                        phys_h_t)
            decoder_frames.append(output_image)

        encoder_frames = torch.stack(encoder_frames, dim=1)
        decoder_frames = torch.stack(decoder_frames, dim=1)
        phys_filter_encoder = self.encoder.phycell.cell_list[0].F.conv1.weight
        phys_filter_decoder = self.decoder.phycell.cell_list[0].F.conv1.weight

        return encoder_frames, decoder_frames, phys_filter_encoder, phys_filter_decoder



class EncoderRNN(pl.LightningModule):
    def __init__(self, num_channels_in, cnn_hidden_size, num_channels_out, rnn_cell_height, rnn_cell_width,
                 rnn_input_dim,
                 phycell_hidden_dims, kernel_size_phycell, convlstm_hidden_dims, kernel_size_convlstm, downscale=4):

        super(EncoderRNN, self).__init__()
        if downscale == 4:
            self.encoder_E = encoder_4E(nchannels_in=num_channels_in,
                                        nchannels_out=cnn_hidden_size)  # general encoder 64x64x1 -> 32x32x32
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
            raise ("the downscale must in [4, 16, 30, 32]!")

        self.decoder_att = nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=1, stride=1,
                                   padding=0)

        self.encoder_Ep = encoder_specific(nchannels_in=cnn_hidden_size,
                                           nchannels_out=cnn_hidden_size)  # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.decoder_Dp = decoder_specific(nchannels_in=cnn_hidden_size,
                                           nchannels_out=cnn_hidden_size)  # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)

        self.phycell = PhyCell(input_shape=(rnn_cell_height, rnn_cell_width), input_dim=rnn_input_dim,
                               F_hidden_dims=phycell_hidden_dims,
                               kernel_size=(kernel_size_phycell, kernel_size_phycell))
        self.convcell = ConvLSTM(input_shape=(rnn_cell_height, rnn_cell_width), input_dim=rnn_input_dim,
                                 hidden_dims=convlstm_hidden_dims,
                                 kernel_size=(kernel_size_convlstm, kernel_size_convlstm))

    def forward(self, input_, first_timestep=False, h_t=None, c_t=None, phys_h_t=None):
        input_ = self.encoder_E(input_)

        input_phys = self.encoder_Ep(input_)
        input_conv = self.encoder_Er(input_)

        phys_h_t, output1 = self.phycell(input_phys, first_timestep, phys_h_t)
        (h_t, c_t), output2 = self.convcell(input_conv, first_timestep, h_t, c_t)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr
        output_image = torch.sigmoid(self.decoder_D(concat))
        att_image = self.decoder_att(concat)
        return h_t, c_t, phys_h_t, out_phys, out_conv, output_image, att_image


class DecoderRNN_ATT(pl.LightningModule):
    def __init__(self, num_channels_in, cnn_hidden_size, num_channels_out, rnn_cell_height, rnn_cell_width,
                 rnn_input_dim, phycell_hidden_dims, kernel_size_phycell, convlstm_hidden_dims, kernel_size_convlstm,
                 downscale=4, input_length=6):

        super(DecoderRNN_ATT, self).__init__()
        if downscale == 4:
            self.encoder_E = encoder_4E(nchannels_in=num_channels_in,
                                        nchannels_out=cnn_hidden_size)  # general encoder 64x64x1 -> 32x32x32
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
            raise ("the downscale must in [4, 16, 30, 32]!")

        self.encoder_Ep = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)  # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.decoder_Dp = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)  # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)

        self.phycell = PhyCell(input_shape=(rnn_cell_height, rnn_cell_width), input_dim=rnn_input_dim,
                          F_hidden_dims=phycell_hidden_dims, kernel_size=(kernel_size_phycell, kernel_size_phycell))
        self.convcell = ConvLSTM(input_shape=(rnn_cell_height, rnn_cell_width), input_dim=rnn_input_dim,
                            hidden_dims=convlstm_hidden_dims, kernel_size=(kernel_size_convlstm, kernel_size_convlstm))

        self.att = nn.Conv2d(in_channels=rnn_input_dim, out_channels=input_length, kernel_size=1, stride=1, padding=0)
        self.att_combine = nn.Conv2d(in_channels=2*rnn_input_dim, out_channels=rnn_input_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, input_, output_seqs, first_timestep=False, h_t=None, c_t=None, phys_h_t=None):

        input_ = self.encoder_E(input_)
        att_weight = F.softmax(self.att(input_), dim=1)
        att_applied = torch.sum(att_weight.unsqueeze(2) * output_seqs, dim=1)
        input_ = self.att_combine(torch.cat([input_, att_applied], dim=1))
        input_phys = self.encoder_Ep(input_)
        input_conv = self.encoder_Er(input_)

        phys_h_t, output1 = self.phycell(input_phys, first_timestep, phys_h_t)
        (h_t, c_t), output2 = self.convcell(input_conv, first_timestep, h_t, c_t)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr
        output_image = self.decoder_D(concat)
        return h_t, c_t, phys_h_t, out_phys, out_conv, output_image


class PhyCell_Cell(pl.LightningModule):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2',
                          nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)  # prediction
        next_hidden = hidden_tilde + K * (x - hidden_tilde)  # correction , Haddamard product
        return next_hidden


class PhyCell(pl.LightningModule):
    def __init__(self, input_shape, input_dim, F_hidden_dims, kernel_size):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = len(F_hidden_dims)
        self.kernel_size = kernel_size
        self.H = []

        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False, h_t=None):  # input_ [batch_size, 1, channels, width, height]
        # batch_size = input_.data.size()[0]
        if first_timestep:
            #   self.initHidden(batch_size)  # init Hidden at each forward start
            self.H = h_t

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])

        return self.H, self.H


class ConvLSTM_Cell(pl.LightningModule):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(pl.LightningModule):
    def __init__(self, input_shape, input_dim, hidden_dims, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        self.kernel_size = kernel_size
        self.H, self.C = [], []

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False, h_t=None,
                c_t=None):  # input_ [batch_size, 1, channels, width, height]
        if first_timestep:
            self.H = h_t
            self.C = c_t

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H  # (hidden, output)


class dcgan_conv(pl.LightningModule):
    def __init__(self, channels_in, channels_out, stride, kernel_size=3, padding=1):
        super(dcgan_conv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.GroupNorm(16, channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.down_conv(input_)


class dcgan_upconv(pl.LightningModule):
    def __init__(self, channels_in, channels_out, stride, kernel_size=3, padding=1, output_padding=0):
        super(dcgan_upconv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, output_padding=output_padding),
            nn.GroupNorm(16, channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.up_conv(input_)


class encoder_4E(pl.LightningModule):
    def __init__(self, nchannels_in=1, nchannels_out=64):
        super(encoder_4E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out, stride=2)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3

class encoder_16E(pl.LightningModule):
    def __init__(self, nchannels_in=1, nchannels_out=128):
        super(encoder_16E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c4 = dcgan_conv(nchannels_out // 2, nchannels_out, stride=2)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        return h4

class decoder_16D(pl.LightningModule):
    def __init__(self, nchannels_in=128, nchannels_out=1):
        super(decoder_16D, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc3 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc4 = nn.ConvTranspose2d(in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=(3, 3),
                                       stride=2, padding=1, output_padding=1)  # (nc) x 64 x 64

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        return d4


class encoder_30E(pl.LightningModule):
    def __init__(self, nchannels_in=1, nchannels_out=64):
        super(encoder_30E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, kernel_size=7, stride=5, padding=1)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, kernel_size=5, stride=3,
                             padding=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out, kernel_size=3, stride=2, padding=1)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_4D(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=1):
        super(decoder_4D, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=(3, 3),
                                       stride=2, padding=1, output_padding=1)  # (nc) x 64 x 64

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class decoder_30D(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=1):
        super(decoder_30D, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, kernel_size=4, stride=2, padding=1)
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, kernel_size=5, stride=3, padding=1)
        self.upc3 = nn.ConvTranspose2d(in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=7, \
                                       stride=5, padding=1)

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3

class encoder_32E(pl.LightningModule):
    def __init__(self, nchannels_in = 1, nchannels_out=128):
        super(encoder_32E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nchannels_in, nchannels_out//2, kernel_size=5, stride=4, padding=1)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nchannels_out//2, nchannels_out//2, kernel_size=3, stride=2, padding=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nchannels_out//2, nchannels_out, kernel_size=5, stride=4, padding=1)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3

class decoder_32D(pl.LightningModule):
    def __init__(self, nchannels_in=128, nchannels_out=1):
        super(decoder_32D, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in//2, kernel_size=6, stride=4, padding=1)
        self.upc2 = dcgan_upconv(nchannels_in//2, nchannels_in//2, kernel_size=4, stride=2, padding=1)
        self.upc3 = nn.ConvTranspose2d(in_channels=nchannels_in//2, out_channels=nchannels_out, kernel_size=6,\
                                       stride=4, padding=1)

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nchannels_in, nchannels_out, stride=1)  # (64) x 16 x 16
        self.c2 = dcgan_conv(nchannels_out, nchannels_out, stride=1)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        return h2


class decoder_specific(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in, stride=1)  # (64) x 16 x 16
        self.upc2 = dcgan_upconv(nchannels_in, nchannels_out, stride=1)  # (32) x 32 x 32

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        return d2


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = CGsNet.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    m = CGsNet(**dict_args)
    x = torch.randn(7, 10, 1, 900, 1200)
    y = m(x, None, False)