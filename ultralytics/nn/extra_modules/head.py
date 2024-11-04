import math
import torch
import torch.nn as nn
from ultralytics.nn.extra_modules.block import ACmix
from ultralytics.nn.extra_modules.rep_block import DiverseBranchBlock
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.utils.checks import check_version
from ultralytics.utils.tal import dist2bbox, make_anchors, dist2rbox
from ..modules import DFL


########################################  标准v5检测头 ########################################
class v5Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m = nn.ModuleList(DoubleConv(x, self.no * self.na, self.nc) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x:torch.Tensor):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
########################################  标准v5检测头 ########################################


######################################## ACDetect ########################################
class ACDetect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc + 1  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.cls = nn.ModuleList(Conv(x, self.nc * self.na, 1) for x in ch)
        self.loc = nn.ModuleList(Conv(x, 4 * self.na, 1) for x in ch)
        self.pre_ac = nn.ModuleList(Conv(x, self.no * self.na, 1) for x in ch)
        self.ac = nn.ModuleList(ACmix(self.no * self.na, self.no * self.na, (4, 4), 8) for x in ch)
        self.ac_cls = nn.ModuleList(nn.Conv2d(self.no * self.na, self.nc * self.na, 1) for x in ch)
        self.ac_loc = nn.ModuleList(nn.Conv2d(self.no * self.na, 4 * self.na, 1) for x in ch)
        self.cls2 = nn.ModuleList(Conv(2 * self.nc * self.na, self.nc * self.na, 1) for x in ch)
        self.loc2 = nn.ModuleList(Conv(2 * 4 * self.na, 4 * self.na, 1) for x in ch)
        self.cls_anchor1 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.cls_anchor2 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.cls_anchor3 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.cls_anchor4 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.loc_anchor1 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.loc_anchor2 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.loc_anchor3 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.loc_anchor4 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x_cls = self.cls[i](x[i])
            x_ac = self.ac[i](self.pre_ac[i](x[i]))
            x_loc = self.loc[i](x[i])
            ac_cls = self.ac_cls[i](x_ac)
            ac_loc = self.ac_loc[i](x_ac)
            x_cls = self.cls2[i](torch.cat((x_cls, ac_cls), dim=1))
            x_loc = self.loc2[i](torch.cat((x_loc, ac_loc), dim=1))
            x_cls_1 = self.cls_anchor1[i](x_cls)
            x_cls_2 = self.cls_anchor2[i](x_cls)
            x_cls_3 = self.cls_anchor3[i](x_cls)
            x_cls_4 = self.cls_anchor4[i](x_cls)
            x_loc_1 = self.loc_anchor1[i](x_loc)
            x_loc_2 = self.loc_anchor2[i](x_loc)
            x_loc_3 = self.loc_anchor3[i](x_loc)
            x_loc_4 = self.loc_anchor4[i](x_loc)

            x[i] = torch.cat([x_loc_1, x_cls_1, x_loc_2, x_cls_2, x_loc_3, x_cls_3, x_loc_4, x_cls_4], dim=1)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
######################################## ACDetect ########################################


class STDetect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc + 1  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.cls = nn.ModuleList(Conv(int(x / 2), self.nc * self.na, 1) for x in ch)
        self.loc = nn.ModuleList(Conv(int(x / 2), 4 * self.na, 1) for x in ch)
        self.cls_anchor1 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.cls_anchor2 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.cls_anchor3 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.cls_anchor4 = nn.ModuleList(nn.Conv2d(self.nc * self.na, self.nc, 1) for x in ch)
        self.loc_anchor1 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.loc_anchor2 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.loc_anchor3 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.loc_anchor4 = nn.ModuleList(nn.Conv2d(4 * self.na, 4, 1) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            b, c, h, w = x[i].shape
            x_cls = x[i][:, :int(c / 2), :, :]
            x_loc = x[i][:, int(c / 2):, :, :]
            x_cls = self.cls[i](x_cls)
            x_loc = self.loc[i](x_loc)
            x_cls_1 = self.cls_anchor1[i](x_cls)
            x_cls_2 = self.cls_anchor2[i](x_cls)
            x_cls_3 = self.cls_anchor3[i](x_cls)
            x_cls_4 = self.cls_anchor4[i](x_cls)
            x_loc_1 = self.loc_anchor1[i](x_loc)
            x_loc_2 = self.loc_anchor2[i](x_loc)
            x_loc_3 = self.loc_anchor3[i](x_loc)
            x_loc_4 = self.loc_anchor4[i](x_loc)

            x[i] = torch.cat([x_loc_1, x_cls_1, x_loc_2, x_cls_2, x_loc_3, x_cls_3, x_loc_4, x_cls_4], dim=1)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid





######################################## 重参数轻量化检测头 Detect_RSCD ########################################
class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
    
class Detect_LSCD(nn.Module):
    # Lightweight Shared Convolutional Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3), Conv_GN(hidc, hidc, 3))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

class Detect_RSCD(Detect_LSCD):
    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.share_conv = nn.Sequential(DiverseBranchBlock(hidc, hidc, 3), DiverseBranchBlock(hidc, hidc, 3))
        # self.share_conv = nn.Sequential(DeepDiverseBranchBlock(hidc, hidc, 3), DeepDiverseBranchBlock(hidc, hidc, 3))
        # self.share_conv = nn.Sequential(WideDiverseBranchBlock(hidc, hidc, 3), WideDiverseBranchBlock(hidc, hidc, 3))
        # self.share_conv = nn.Sequential(RepConv(hidc, hidc, 3), RepConv(hidc, hidc, 3))

######################################## 重参数轻量化检测头 Detect_RSCD ########################################