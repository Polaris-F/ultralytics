
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv, autopad
from ..modules.block import *

from timm.models.layers import trunc_normal_

# from .attention import *
# from .rep_block import *
# from .kernel_warehouse import KWConv
# from .dynamic_snake_conv import DySnakeConv
# from .ops_dcnv3.modules import DCNv3, DCNv3_DyHead
# from .shiftwise_conv import ReparamLargeKernelConv
# from .mamba_vss import *
# from .fadc import AdaptiveDilatedConv
# from .hcfnet import PPA
# from ..backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite
# from ..backbone.rmt import RetBlock, RelPos2d
# from .kan_convs import FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer
# from .deconv import DEConv
# from .SMPConv import SMPConv
# from .camixer import CAMixer
# from .orepa import *
# from .RFAconv import *
# from .wtconv2d import *
# from .metaformer import *
# from .tsdn import DTAB, LayerNorm

# from ultralytics.utils.torch_utils import make_divisible


__all__ = (
    "ACmix","SPDConv","CSPOmniKernel",
    "AnaC2f"
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

######################################## SPD-Conv start ########################################

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

######################################## SPD-Conv end ########################################


######################################## Omni-Kernel Network for Image Restoration [AAAI-24] start ########################################

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(self.cv1.conv.out_channels * self.e), int(self.cv1.conv.out_channels * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

######################################## Omni-Kernel Network for Image Restoration [AAAI-24] end ########################################


######################################## AnaC2f By Analogical reasoning module [AAAI-25] start ########################################


class ScoreCompute(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale
        aw_single = aw.mean(dim=1,keepdim=True)
        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w), aw_single.squeeze(1)

class AnaC2f(nn.Module):
    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
       
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = ScoreCompute(self.c, self.c, gc=gc, ec=ec, nh=nh)
        self.graph_update = GraphUpdate(c2, 64, c2)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        new,score = self.attn(y[-1], guide)
        y.append(new)
        z = self.cv2(torch.cat(y,1))
        z = update_features_with_gcn(z, score, self.graph_update, k_ratio=0.007,similarity_threshold=0.5)
        return z

from torch_geometric.nn import GCNConv
class GraphUpdate(nn.Module):
    """Graph-based feature update module."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)

    # def forward(self, x, edge_index):
    #     """Forward pass for graph convolution."""
    #     # print(x.device,edge_index.device)
    #     edge_index=edge_index.to(x.device)
    #     q = self.gcn1(x, edge_index)
    #     x = F.relu(q)
    #     x = self.gcn2(x, edge_index)
    #     return x
    def forward(self, x, edge_index, edge_weight=None):
        edge_index = edge_index.to(x.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(x.device)
        
        q = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(q)
        x = self.gcn2(x, edge_index, edge_weight)
        return x
def select_top_k_pixels(score, k_ratio=0.04):
    batch_size, height, width = score.shape
    num_pixels = height * width
    num_select = int(num_pixels * k_ratio)

    # Flatten the score map
    flat_score = score.view(batch_size, -1)  # [batch, height * width]

    # Get top k% indices
    _, top_indices = torch.topk(flat_score, num_select, dim=1)  # [batch, num_select]

    # Convert flat indices to 2D (h, w) coordinates
    h_indices = top_indices // width
    w_indices = top_indices % width
    top_indices_2d = torch.stack([h_indices, w_indices], dim=-1)  # [batch, num_select, 2]

    return top_indices_2d, top_indices

def build_edge_index_from_features_v1(features, threshold=0.55):
    # Compute cosine similarity between features
    norm_features = F.normalize(features, p=2, dim=1)  # L2 normalization
    similarity_matrix = torch.mm(norm_features, norm_features.t())  # [num_nodes, num_nodes]
    # Build edges based on similarity threshold
    edges = torch.nonzero(similarity_matrix > threshold, as_tuple=False).t()  # [2, num_edges]
    if edges.size(1) == 0:
        num_nodes = features.size(0)
        self_loops = torch.arange(num_nodes).repeat(2, 1)  # [2, num_nodes]
        edges = self_loops

    return edges

def build_edge_index_from_features_v2(features, threshold=0.7, max_edges_per_node=10):
    """构建特征间边索引，使用动态批大小以优化内存使用"""
    num_nodes = features.size(0)
    feature_dim = features.size(1)
    
    # 动态计算批次大小
    # 较小数据集用大批次，大数据集用小批次
    if num_nodes <= 100:
        batch_size = num_nodes  # 小数据集全量处理
    elif num_nodes <= 1000:
        batch_size = num_nodes // 5  # 中等数据集
    else:
        # 大数据集，根据节点数和特征维度估算适当批次
        memory_factor = max(1, min(20, int(2048 / feature_dim)))
        batch_size = max(32, num_nodes // memory_factor)
    
    # 确保批次大小合理
    batch_size = max(1, min(batch_size, num_nodes))
    
    edges_list = []
    for i in range(0, num_nodes, batch_size):
        end_idx = min(i + batch_size, num_nodes)
        batch_features = features[i:end_idx]
        
        # 计算当前批次与所有特征的相似度
        norm_batch = F.normalize(batch_features, p=2, dim=1)
        norm_all = F.normalize(features, p=2, dim=1)
        batch_sim = torch.mm(norm_batch, norm_all.t())  # [batch, num_nodes]
        
        # 只保留每个节点最相似的K个边
        topk = min(max_edges_per_node, num_nodes)
        _, top_indices = torch.topk(batch_sim, topk, dim=1)
        
        # 构建边索引
        for j, node_idx in enumerate(range(i, end_idx)):
            for neighbor_idx in top_indices[j]:
                neighbor_idx = neighbor_idx.item()
                if batch_sim[j, neighbor_idx] > threshold and node_idx != neighbor_idx:
                    edges_list.append([node_idx, neighbor_idx])
    
    if len(edges_list) == 0:
        # 如果没有边，创建自环
        self_loops = torch.arange(num_nodes, device=features.device).repeat(2, 1)
        return self_loops
    
    edges = torch.tensor(edges_list, device=features.device).t()
    return edges

def build_edge_index_from_features_with_weights(features, threshold=0.55):
    # Compute cosine similarity between features
    norm_features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.mm(norm_features, norm_features.t())
    
    # Find edges above threshold
    mask = similarity_matrix > threshold
    edge_index = torch.nonzero(mask, as_tuple=False).t()
    
    # Extract weights from similarity matrix
    edge_weight = similarity_matrix[mask]
    
    if edge_index.size(1) == 0:
        num_nodes = features.size(0)
        edge_index = torch.arange(num_nodes).repeat(2, 1)
        edge_weight = torch.ones(num_nodes, device=features.device)
    
    return edge_index, edge_weight
def update_features_with_gcn(z, score, graph_update_module, k_ratio=0.001, similarity_threshold=0.6):

    batch_size, channels, height, width = z.shape
    # Select top k% pixels
    top_indices_2d, flat_indices = select_top_k_pixels(score, k_ratio)
    # Extract features of top pixels
    flat_z = z.view(batch_size, channels, -1)  # [batch, channel, height * width]
    top_features = flat_z.gather(2, flat_indices.unsqueeze(1).expand(-1, channels, -1))  # [batch, channel, num_select]
    top_features = top_features.permute(0, 2, 1).reshape(-1, channels) # [batch * num_select, channel]
    # Build edge index based on feature similarity

    ### 使用边的权重信息
    # edge_index, edge_weight = build_edge_index_from_features_with_weights(top_features, similarity_threshold)
    # Apply graph convolution
    # updated_features = graph_update_module(top_features, edge_index, edge_weight)  # [batch * num_select, channel]
    
    ### 不用边的权重信息
    edge_index = build_edge_index_from_features_v1(top_features, similarity_threshold)
    updated_features = graph_update_module(top_features, edge_index)  # [batch * num_select, channel]

    # Reshape updated features
    updated_features = updated_features.view(batch_size, -1, channels).permute(0, 2, 1)  # [batch, channel, num_select]
    # Replace original features with updated features
    new_z = z.clone()
    updated_features = updated_features.to(new_z.dtype)
    new_z.view(batch_size, channels, -1).scatter_(2, flat_indices.unsqueeze(1).expand(-1, channels, -1), updated_features)
    return new_z
