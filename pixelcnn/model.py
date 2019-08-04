import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import MaskedConv2d, CroppedConv2d

from utils import subdict


class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, data_channels):
        super(CausalConv2d, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = MaskedConv2d(2 * out_channels,
                                   2 * out_channels,
                                   (1, 1),
                                   mask_type=mask_type,
                                   data_channels=data_channels)

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type=mask_type,
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type=mask_type,
                                 data_channels=data_channels)

    def forward(self, x):
        v_in, h_in = x[0], x[1]

        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_gate = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)

        h_out = self.h_fc(h_gate)

        return {0: v_out, 1: h_out, 2: h_gate}


class GatedBlock(CausalConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, data_channels):
        super(GatedBlock, self).__init__(in_channels, out_channels, kernel_size, mask_type, data_channels)

        self.h_skip = MaskedConv2d(out_channels,
                                   out_channels,
                                   (1, 1),
                                   mask_type=mask_type,
                                   data_channels=data_channels)

    def forward(self, x):
        v_in, h_in, skip = x[0], x[1], x[2]

        # run v and h through CasualConv's forward
        x = super(GatedBlock, self).forward(subdict(x, [0, 1]))
        v_out, h_out, h_gate = x[0], x[1], x[2]

        # skip connection
        skip = skip + self.h_skip(h_gate)

        # residual connection
        h_out = h_out + h_in

        return {0: v_out, 1: h_out, 2: skip}


class PixelCNN(nn.Module):
    def __init__(self, cfg):
        super(PixelCNN, self).__init__()

        self.hidden_fmaps = cfg.hidden_fmaps

        self.color_levels = cfg.color_levels

        self.causal_conv = CausalConv2d(cfg.data_channels,
                                        cfg.hidden_fmaps,
                                        cfg.causal_ksize,
                                        mask_type='A',
                                        data_channels=cfg.data_channels)

        self.hidden_conv = nn.Sequential(
            *[GatedBlock(cfg.hidden_fmaps,
                         cfg.hidden_fmaps,
                         cfg.hidden_ksize,
                         mask_type='B',
                         data_channels=cfg.data_channels) for _ in range(cfg.hidden_layers)],
        )

        self.out_hidden_conv = MaskedConv2d(cfg.hidden_fmaps,
                                            cfg.out_hidden_fmaps,
                                            (1, 1),
                                            mask_type='B',
                                            data_channels=cfg.data_channels)

        self.out_conv = MaskedConv2d(cfg.out_hidden_fmaps,
                                     cfg.data_channels * cfg.color_levels,
                                     (1, 1),
                                     mask_type='B',
                                     data_channels=cfg.data_channels)

    def forward(self, x):
        count, data_channels, height, width = x.size()

        v, h, _ = self.causal_conv({0: x, 1: x}).values()

        _, _, out = self.hidden_conv({0: v, 1: h, 2: x.new_zeros((count, self.hidden_fmaps, height, width), requires_grad=True)}).values()

        out = F.relu(out)
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)

        out = out.view(count, self.color_levels, data_channels, height, width)

        return out

    def sample(self, shape, count, device='cuda'):
        channels, height, width = shape

        samples = torch.zeros(count, *shape).to(device)

        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        unnormalized_probs = self.forward(samples)
                        pixel_probs = torch.softmax(unnormalized_probs[:, :, c, i, j], dim=1)
                        sampled_levels = torch.multinomial(pixel_probs, 1).squeeze() / (self.color_levels - 1)
                        samples[:, c, i, j] = sampled_levels

        return samples
