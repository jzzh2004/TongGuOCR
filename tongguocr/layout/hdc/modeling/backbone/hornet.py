# Copyright (c) 2021 Yongming Rao. MIT; see THIRD_PARTY_LICENSES/HorNet-MIT.md.
# Adapted for TongGuOCR's detection backbone.
from functools import partial
from mmdet.utils import get_root_logger
from mmcv.runner import _load_checkpoint
from timm.models.layers import trunc_normal_, DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import warnings
import math
import os
import sys
import torch.fft
import traceback
import torch.utils.checkpoint as checkpoint

if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM

        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)

        print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())

        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim,
                             dim,
                             kernel_size=kernel,
                             padding=(kernel - 1) // 2,
                             bias=bias,
                             groups=dim)

        print(
            '[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl'
        )
else:

    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim,
                         dim,
                         kernel_size=kernel,
                         padding=(kernel - 1) // 2,
                         bias=bias,
                         groups=dim)

    print('Using PyTorch large kernel dw conv impl')


class GlobalLocalFilter(nn.Module):

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2,
                            dim // 2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            groups=dim // 2)
        self.complex_weight = nn.Parameter(
            torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight.to(torch.float32)
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2),
                                   size=x2.shape[2:4],
                                   mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        # import pdb; pdb.set_trace()
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho').to(x.dtype)

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)],
                      dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnconv(nn.Module):

    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], 1)
            for i in range(order - 1)
        ])

        self.scale = s

        print('[gconv]', order, 'order with dims=', self.dims,
              'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self,
                 dim,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim,
            4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1

        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))
        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class HorNet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 depths=[3, 3, 9, 3],
                 base_dim=96,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 gnconv=gnconv,
                 block=Block,
                 out_indices=(0, 1, 2, 3),
                 num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(HorNet, self).__init__()
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        self.out_indices = out_indices
        self.num_stages = num_stages
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        self.downsample_layers = nn.ModuleList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList(
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                block(dim=dims[i],
                      drop_path=dp_rates[cur + j],
                      layer_scale_init_value=layer_scale_init_value,
                      gnconv=gnconv[i]) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.init_weights()

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(self.init_cfg.checkpoint,
                                    logger=logger,
                                    map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            # load state_dict
            self.load_state_dict(_state_dict, False)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return tuple(outs)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class HorNet_tiny_7x7(HorNet):

    def __init__(self,
                 depths=[2, 3, 18, 2],
                 base_dim=64,
                 block=Block,
                 drop_path_rate=0.2,
                 gnconv=[
                     partial(gnconv, order=2, s=1.0 / 3.0),
                     partial(gnconv, order=3, s=1.0 / 3.0),
                     partial(gnconv, order=4, s=1.0 / 3.0),
                     partial(gnconv, order=5, s=1.0 / 3.0),
                 ],
                 pretrained=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(HorNet_tiny_7x7, self).__init__(depths=depths,
                                              drop_path_rate=drop_path_rate,
                                              base_dim=base_dim,
                                              gnconv=gnconv,
                                              block=block,
                                              pretrained=pretrained,
                                              init_cfg=init_cfg,
                                              *args,
                                              **kwargs)


class HorNet_tiny_gf(HorNet):

    def __init__(self,
                 depths=[2, 3, 18, 2],
                 base_dim=64,
                 block=Block,
                 drop_path_rate=0.4,
                 gnconv=[
                     partial(gnconv, order=2, s=1.0 / 3.0),
                     partial(gnconv, order=3, s=1.0 / 3.0),
                     partial(gnconv,
                             order=4,
                             s=1.0 / 3.0,
                             h=14,
                             w=8,
                             gflayer=GlobalLocalFilter),
                     partial(gnconv,
                             order=5,
                             s=1.0 / 3.0,
                             h=7,
                             w=4,
                             gflayer=GlobalLocalFilter),
                 ],
                 pretrained=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(HorNet_tiny_gf, self).__init__(depths=depths,
                                             base_dim=base_dim,
                                             drop_path_rate=drop_path_rate,
                                             gnconv=gnconv,
                                             block=block,
                                             pretrained=pretrained,
                                             init_cfg=init_cfg,
                                             *args,
                                             **kwargs)
