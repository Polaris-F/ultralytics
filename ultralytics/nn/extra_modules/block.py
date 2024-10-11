# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

__all__ = (
    "ACmix",
)


## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ACmix >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ##

def ones(tensor:torch.Tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def zeros(tensor:torch.Tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

def window_partition(x:torch.Tensor, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class ACmix(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, c1, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))   # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # fully connected layer in Fig.2
        self.fc:torch.Tensor = nn.Conv2d(3 * self.num_heads, 9, kernel_size=1, bias=True)
        # group convolution layer in Fig.3
        self.dep_conv = nn.Conv2d(9 * dim // self.num_heads, dim, kernel_size=3, bias=True,
                                  groups=dim // self.num_heads, padding=1)
        # rates for both paths
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.rate1)
        ones(self.rate2)
        # shift initialization for group convolution
        kernel = torch.zeros(9, 3, 3)
        for i in range(9):
            kernel[i, i // 3, i % 3] = 1.
        kernel = kernel.squeeze(0).repeat(self.dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = zeros(self.dep_conv.bias)

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: input features with shape of (B, C, H, W) To (B, H, W, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        _, H, W, _ = x.shape
        qkv:torch.Tensor = self.qkv(x)

        # fully connected layer
        f_all = qkv.reshape(x.shape[0], H * W, 3 * self.num_heads, -1).permute(0, 2, 1, 3).contiguous()  # B, 3*nhead, H*W, C//nhead
        fc_out:torch.Tensor = self.fc(f_all)
        f_conv = fc_out.permute(0, 3, 1, 2).contiguous().reshape(x.shape[0], 9 * x.shape[-1] // self.num_heads, H, W)

        # group conovlution
        out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 1).contiguous()  # B, H, W, C

        # partition windows
        qkv = window_partition(qkv, self.window_size[0])  # nW*B, window_size, window_size, C

        B_, _, _, C = qkv.shape

        qkv = qkv.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        N = self.window_size[0] * self.window_size[1]
        C = C // 3

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size[0], H, W)  # B H' W' C

        x = self.rate1 * x + self.rate2 * out_conv

        x = self.proj_drop(x).permute(0, 3, 1, 2).contiguous()
        return x

## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ACmix <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ##
