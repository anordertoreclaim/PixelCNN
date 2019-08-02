import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import MaskedConv2d, CroppedConv2d


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, data_channels, residual=True):
        super(GatedBlock, self).__init__()
        self.split_index = out_channels
        self.residual = residual

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
                                   data_channels=data_channels, )
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
        self.h_skip = MaskedConv2d(out_channels,
                                   out_channels,
                                   (1, 1),
                                   mask_type=mask_type,
                                   data_channels=data_channels)

    def forward(self, x):
        v_in, h_in, skip = x[0], x[1], x[2]

        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out = torch.tanh(v_out[:, :self.split_index]) * torch.sigmoid(v_out[:, self.split_index:])

        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out = torch.tanh(h_out[:, :self.split_index]) * torch.sigmoid(h_out[:, self.split_index:])

        if skip is None:
            skip = self.h_skip(h_out)
        else:
            skip = skip + self.h_skip(h_out)

        h_out = self.hc(h_out)

        if self.residual:
            h_out = h_out + h_in

        return {0: v_out, 1: h_out, 2: skip}


class PixelCNN(nn.Module):
    def __init__(self, cfg):
        super(PixelCNN, self).__init__()
        self.causal_ksize = cfg.causal_ksize
        self.hidden_ksize = cfg.hidden_ksize

        self.data_channels = cfg.data_channels
        self.hidden_fmaps = cfg.hidden_fmaps
        self.out_hidden_fmaps = cfg.out_hidden_fmaps

        self.hidden_layers = cfg.hidden_layers

        self.color_levels = cfg.color_levels

        self.causal_conv = GatedBlock(self.data_channels,
                                      self.hidden_fmaps,
                                      self.causal_ksize,
                                      mask_type='A',
                                      data_channels=self.data_channels,
                                      residual=False)

        self.hidden_conv = nn.Sequential(
            *[GatedBlock(self.hidden_fmaps,
                         self.hidden_fmaps,
                         self.hidden_ksize,
                         mask_type='B',
                         data_channels=self.data_channels) for _ in range(self.hidden_layers)],
        )

        self.out_hidden_conv = MaskedConv2d(self.hidden_fmaps,
                                            self.out_hidden_fmaps,
                                            (1, 1),
                                            mask_type='B',
                                            data_channels=self.data_channels)

        self.out_conv = MaskedConv2d(self.out_hidden_fmaps,
                                     self.data_channels * self.color_levels,
                                     (1, 1),
                                     mask_type='B',
                                     data_channels=self.data_channels,
                                     out_spread=False)

    def forward(self, x):
        count, _, height, width = x.size()

        v, h, _ = self.causal_conv({0: x, 1: x, 2: None}).values()

        _, _, out = self.hidden_conv({0: v, 1: h, 2: None}).values()

        out = F.relu(out)
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)

        batch_size, _, height, width = out.size()
        out = out.view(batch_size, self.color_levels, self.data_channels, height, width)

        return out

    def sample(self, shape, count):
        channels, height, width = shape

        samples = torch.zeros(count, *shape)

        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        unnormalized_probs = self.forward(samples)
                        pixel_probs = torch.softmax(unnormalized_probs[:, :, c, i, j], dim=1)
                        sampled_levels = torch.multinomial(pixel_probs, 1).squeeze()
                        samples[:, c, i, j] = sampled_levels

        return samples
