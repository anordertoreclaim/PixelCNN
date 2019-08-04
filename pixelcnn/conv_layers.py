import torch
import torch.nn as nn

import numpy as np


class CroppedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(CroppedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super(CroppedConv2d, self).forward(x)

        kernel_height, _ = self.kernel_size
        res = x[:, :, 1:-kernel_height, :]
        shifted_up_res = x[:, :, :-kernel_height-1, :]

        return res, shifted_up_res


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type, data_channels, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B'], 'Invalid mask type.'

        out_channels, in_channels, height, width = self.weight.size()
        yc, xc = height // 2, width // 2

        mask = np.zeros(self.weight.size(), dtype=np.float32)
        mask[:, :, :yc, :] = 1
        mask[:, :, yc, :xc + 1] = 1

        def cmask(out_c, in_c):
            a = (np.arange(out_channels) % data_channels == out_c)[:, None]
            b = (np.arange(in_channels) % data_channels == in_c)[None, :]
            return a * b

        for o in range(data_channels):
            for i in range(o + 1, data_channels):
                mask[cmask(o, i), yc, xc] = 0

        if mask_type == 'A':
            for c in range(data_channels):
                mask[cmask(c, c), yc, xc] = 0

        mask = torch.from_numpy(mask).float()

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(MaskedConv2d, self).forward(x)
        return x
