import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nowcastnet.layers.utils import warp, make_grid
from nowcastnet.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from nowcastnet.layers.evolution.evolution_network import Evolution_Network
from nowcastnet.layers.generation.noise_projector import Noise_Projector

class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.configs = configs
        self.output_sequence_length = self.configs.total_sequence_length - self.configs.input_sequence_length

        self.evo_net = Evolution_Network(self.configs.input_sequence_length, self.output_sequence_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.configs.total_sequence_length, base_c=self.configs.ngf)
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)

        sample_tensor = torch.zeros(1, 1, self.configs.img_size[0], self.configs.img_size[1])
        self.grid = make_grid(sample_tensor)

    def forward(self, all_frames):
        all_frames = all_frames[:, :, :, :, :1] # [B,T,H,W,1], last dime is just for channel

        frames = all_frames.permute(0, 1, 4, 2, 3)  # [B,T,C,H,W]
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.configs.input_sequence_length]   # [B, input_seq_len, C, H, W]
        input_frames = input_frames.reshape(batch, self.configs.input_sequence_length, height, width)   

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)  # [B, output_seq_len, H, W], [B, output_seq_len*2, H, W]
        motion_ = motion.reshape(batch, self.output_sequence_length, 2, height, width)  # [B, output_seq_len, 2, H, W]
        intensity_ = intensity.reshape(batch, self.output_sequence_length, 1, height, width)    # [B, output_seq_len, 1, H, W]
        series = []
        last_frames = all_frames[:, (self.configs.input_sequence_length - 1):self.configs.input_sequence_length, :, :, 0]   # [B, 1, H, W]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.output_sequence_length):
            last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")  # [B, 1, H, W]
            last_frames = last_frames + intensity_[:, i]    # [B, 1, H, W]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        evo_result = evo_result/128
        
        # Generative Network
        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))

        noise = torch.randn(batch, self.configs.ngf, height // 32, width // 32).cuda()
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_result)

        return gen_result.unsqueeze(-1)