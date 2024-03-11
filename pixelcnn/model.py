import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import MaskedConv2d, CroppedConv2d


class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels):
        super(CausalBlock, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = nn.Conv2d(2 * out_channels,
                                2 * out_channels,
                                (1, 1))

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type='A',
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type='A',
                                 data_channels=data_channels)

    def forward(self, image):
        v_out, v_shifted = self.v_conv(image)
        v_out += self.v_fc(image)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(image)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)
        h_out = self.h_fc(h_out)

        return v_out, h_out


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels, classes):
        super(GatedBlock, self).__init__()
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
                                   mask_type='B',
                                   data_channels=data_channels)

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type='B',
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type='B',
                                 data_channels=data_channels)

        self.h_skip = MaskedConv2d(out_channels,
                                   out_channels,
                                   (1, 1),
                                   mask_type='B',
                                   data_channels=data_channels)

        self.label_embedding = nn.Embedding(classes, 2*out_channels)

    def forward(self, x):
        v_in, h_in, skip, label = x[0], x[1], x[2], x[3]

        label_embedded = self.label_embedding(label).unsqueeze(2).unsqueeze(3)

        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out += label_embedded
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out += label_embedded
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)

        # skip connection
        skip = skip + self.h_skip(h_out)

        h_out = self.h_fc(h_out)

        # residual connections
        h_out = h_out + h_in
        v_out = v_out + v_in

        return {0: v_out, 1: h_out, 2: skip, 3: label}


class PixelCNN(nn.Module):
    def __init__(self, cfg):
        super(PixelCNN, self).__init__()

        DATA_CHANNELS = 3
        self.classes = cfg.classes

        self.hidden_fmaps = cfg.hidden_fmaps
        self.color_levels = cfg.color_levels

        self.causal_conv = CausalBlock(DATA_CHANNELS,
                                       cfg.hidden_fmaps,
                                       cfg.causal_ksize,
                                       data_channels=DATA_CHANNELS)

        self.hidden_conv = nn.Sequential(
            *[GatedBlock(cfg.hidden_fmaps, cfg.hidden_fmaps, cfg.hidden_ksize, DATA_CHANNELS, self.classes) for _ in range(cfg.hidden_layers)]
        )

        self.label_embedding = nn.Embedding(self.classes, self.hidden_fmaps)

        self.out_hidden_conv = MaskedConv2d(cfg.hidden_fmaps,
                                            cfg.out_hidden_fmaps,
                                            (1, 1),
                                            mask_type='B',
                                            data_channels=DATA_CHANNELS)

        self.out_conv = MaskedConv2d(cfg.out_hidden_fmaps,
                                     DATA_CHANNELS * cfg.color_levels,
                                     (1, 1),
                                     mask_type='B',
                                     data_channels=DATA_CHANNELS)

    def forward(self, image, label):
        count, data_channels, height, width = image.size()

        v, h = self.causal_conv(image)

        _, _, out, _ = self.hidden_conv({0: v,
                                         1: h,
                                         2: image.new_zeros((count, self.hidden_fmaps, height, width), requires_grad=True),
                                         3: label}).values()

        label_embedded = self.label_embedding(label).unsqueeze(2).unsqueeze(3)

        # add label bias
        out += label_embedded
        out = F.relu(out)
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)

        out = out.view(count, self.color_levels, data_channels, height, width)

        return out

    def sample(self, shape, count, label=None, device='cuda', pbar=True):
        channels, height, width = shape

        samples = torch.zeros(count, *shape).to(device)
        if label is None:
            labels = torch.randint(high=self.classes, size=(count,)).to(device)
        else:
            labels = (label*torch.ones(count)).to(device).long()
        # print("generating with labels",labels)
        if pbar:
            from tqdm import tqdm
            pbar = tqdm(total=height*width*channels, desc="Generating samples: ".ljust(20))
        # Modify this to only do masked pixels
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        unnormalized_probs = self.forward(samples, labels)
                        pixel_probs = torch.softmax(unnormalized_probs[:, :, c, i, j], dim=1)
                        sampled_levels = torch.multinomial(pixel_probs, 1).squeeze().float() / (self.color_levels - 1)
                        samples[:, c, i, j] = sampled_levels
                        if pbar:
                            pbar.update(1)
        if pbar:
            pbar.close()

        return samples

    def inpaint(self, images, masks, labels, device='cuda', pbar=True):
        assert images.shape == masks.shape, "images and masks are diff shapes {} {}".format(images.shape, masks.shape)
        count,channels, height, width = images.shape

        samples = torch.zeros(*images.shape).to(device)
        # print("generating with labels",labels)
        if pbar:
            from tqdm import tqdm
            pbar = tqdm(total=height*width*channels, desc="Generating samples: ".ljust(20))
        # Modify this to only do masked pixels
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    if torch.all(masks[:,:,i,j] < 0.1):
                        samples[:,:,i,j] = images[:,:,i,j]
                        if pbar:
                            pbar.update(channels)
                    else:
                        for c in range(channels):
                            unnormalized_probs = self.forward(samples, labels)
                            pixel_probs = torch.softmax(unnormalized_probs[:, :, c, i, j], dim=1)
                            sampled_levels = torch.multinomial(pixel_probs, 1).squeeze().float() / (self.color_levels - 1)
                            samples[:, c, i, j] = sampled_levels * masks[:, c, i, j] + images[:,c,i,j]*(1-masks[:,c,i,j])
                            if pbar:
                                pbar.update(1)
        if pbar:
            pbar.close()

        return samples
