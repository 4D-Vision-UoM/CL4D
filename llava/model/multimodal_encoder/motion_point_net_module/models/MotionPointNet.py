from typing import List, Optional

import torch
from torch import nn, einsum
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
import logging
import os
import sys
from timm.models.layers import trunc_normal_

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "modules"))

from layers import furthest_point_sample, QueryAndGroup, three_interpolation

# from cpp.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

import utils

POINTS_DIM = 3
CHANNEL_MAP = lambda x: POINTS_DIM + x


class GroupBN2d(nn.Module):
    def __init__(self, channel, length):
        super().__init__()
        self.length = length
        self.norm = nn.BatchNorm3d(channel)

    def unfold_bn(self, x):
        b, c, n, k = x.shape
        x = (
            x.view(b // self.length, self.length, c, n, k)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b, c, n, k).contiguous()
        return x

    def forward(self, inp):
        return self.unfold_bn(inp)


class Conv2D(nn.Module):
    def __init__(self, channel_list, length):
        super().__init__()

        channel_list[0] = CHANNEL_MAP(channel_list[0])
        layers = [
            [
                nn.Conv2d(channel_list[i], channel_list[i + 1], (1, 1), bias=False),
                GroupBN2d(channel_list[i + 1], length),
                nn.LeakyReLU(0.2),
            ]
            for i in range(len(channel_list) - 1)
        ]
        # flatten out the pairs
        layers = [item for sublist in layers for item in sublist]
        self.conv = nn.Sequential(*layers)

    def forward(self, inp):
        return self.conv(inp)


class Conv1D(nn.Module):
    def __init__(self, channel_list):
        super().__init__()

        layers = [
            [
                nn.Conv1d(channel_list[i], channel_list[i + 1], 1, bias=False),
                nn.BatchNorm1d(channel_list[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            for i in range(len(channel_list) - 1)
        ]
        # flatten out the pairs
        layers = [item for sublist in layers for item in sublist]
        self.conv = nn.Sequential(*layers)

    def forward(self, inp):
        return self.conv(inp)


class PointNetSAModuleMSG(nn.Module):
    def __init__(
        self,
        stride: int,
        radius: float,
        nsamples: int,
        length: int,
        channel_list: List[int],
    ):
        super().__init__()
        self.stride = stride

        # build the sampling layer:
        self.sample_fn = furthest_point_sample
        # holder for the grouper and convs (MLPs, \etc)
        self.grouper = QueryAndGroup(radius, nsamples, use_xyz=True, normalize_dp=True)
        self.convs = Conv2D(channel_list, length)

    def shift_point(self, points, stride):  # B, T, ...
        assert stride in [-1, 0, 1]
        if stride == 0:
            return points
        shifted_points = torch.roll(points, stride, dims=1)

        if stride < 0:
            shift_mean = torch.mean(
                shifted_points[:, :-1] - points[:, :-1], dim=1, keepdim=False
            )
            shifted_points[:, -1] = points[:, -1] + shift_mean
        else:
            shifted_points[:, 0] = points[:, 0]

        return shifted_points

    def forward(self, xyz, features, shape):
        """
        xyz: (B, N, 3) tensor of the xyz coordinates of the features
        features: (B, C, N) tensor of the descriptors of the the features
        """
        batch, num_points = xyz.shape[0], xyz.shape[1]

        # get query_xyz (shifted)
        query_xyz = self.shift_point(
            xyz.view(batch // shape[0], shape[0], -1, 3), -1
        ).view(batch, num_points, 3)

        # downsample
        if self.stride > 1:
            idx = self.sample_fn(query_xyz, query_xyz.shape[1] // self.stride).long()
            query_xyz = torch.gather(query_xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # aggregation
        points, feats = self.grouper(query_xyz, xyz, features)
        # feats = self.convs(torch.cat([points, feats], dim=1)).max(-1)[0]
        feats = (
            torch.sigmoid(points[:, -1:])
            * self.convs(torch.cat([points[:, :3], feats], dim=1))
        ).max(-1)[0]

        return query_xyz, feats


class LatentAttn(nn.Module):
    def __init__(self, width, length, head=8, ratio=1.0, num_basis=12, depth=1):
        super().__init__()
        self.width = width
        self.num_basis = num_basis
        self.ratio = ratio
        self.length = length

        # basis
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor(
            [i for i in range(num_basis)], dtype=torch.float
        ).cuda()
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float)
        )
        # latent
        self.head = head
        self.depth = depth
        self.mask_token = nn.Parameter(
            (1 / (width)) * torch.rand(1, width, 1, dtype=torch.float)
        )
        trunc_normal_(self.mask_token, std=0.02)
        # self.pos_emb = nn.Embedding(length, self.width)
        self.encoder_attn = nn.Conv1d(
            self.width, self.width * 3, kernel_size=1, stride=1
        )
        self.decoder_attn = nn.Conv1d(self.width, self.width, kernel_size=1, stride=1)
        # self.drop = nn.Dropout(0.)
        self.fc = Conv1D([self.width, self.width, 3])
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        dim = q.shape[-1]
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k) * (dim**0.5))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def encoder(self, x):
        # x: B C N
        B, C, N = x.shape

        x_tmp = self.encoder_attn(x)
        x_tmp = x_tmp.view(B, 3, C, N).permute(1, 0, 2, 3).contiguous()
        x_tmp = (
            x_tmp.view(3, B, self.head, C // self.head, N)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )  # 3, B, h, N, C/h

        attn = self.self_attn(x_tmp[0], x_tmp[1], x_tmp[2])
        attn = attn.permute(0, 1, 3, 2).contiguous().view(B, C, -1)

        # transition
        attn_modes = self.get_basis(attn)
        attn = self.compl_mul2d(attn_modes, self.weights) + x

        return attn

    def decoder(self, x, attn):
        # x: B C N
        B, C, N = x.shape
        x_init = x
        attn = (
            attn.view(B, self.head, C // self.head, self.length)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        x_que = (
            self.decoder_attn(x)
            .view(B, self.head, C // self.head, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        x = self.self_attn(x_que, attn, attn)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, N) + x_init

        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        x_cos = torch.cos(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def _mask(self, x):
        with torch.no_grad():
            B, C, N = x.shape
            len_keep = int(N * (1 - self.ratio))

            assert len_keep < 1  # mask all

            if len_keep == 0:
                x_masked = self.mask_token.repeat(B, 1, N)
                mask = torch.ones([B, N], dtype=torch.long, device=x.device)
                return x_masked, mask

            noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

            index_sort = torch.argsort(noise, dim=1)
            index_restore = torch.argsort(index_sort, dim=1)

            index_keep = index_sort[:, :len_keep]
            x_keep = torch.gather(
                x, dim=-1, index=index_keep.unsqueeze(1).repeat(1, C, 1)
            )

            mask_token = self.mask_token.repeat(B, 1, N - len_keep)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, N], dtype=torch.long, device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=index_restore)

            x_masked = torch.cat([x_keep, mask_token], dim=-1)
            x_masked = torch.gather(
                x_masked, dim=-1, index=index_restore.unsqueeze(1).repeat(1, C, 1)
            )

        return x_masked, mask

    def forward_feat(self, xt, x_masked):
        for i in range(self.depth):
            # encoder
            xt = self.encoder(xt)  # b, c, token
            # decoder
            x_masked = self.decoder(x_masked, xt)
        return x_masked

    def forward(self, xt, xn):
        # xt -> b, c, t  xn -> b, c, n
        B, C, L = xt.shape
        t = (
            torch.arange(L, dtype=torch.long, device=xt.device)
            .unsqueeze(0)
            .repeat(B, 1)
        )
        # xt = xt + self.pos_emb(t).transpose(1, 2)

        x_masked, _ = self._mask(xn)
        xn_pre = self.forward_feat(xt, x_masked)

        return xn_pre.contiguous(), self.fc(xn_pre).transpose(1, 2).contiguous()


class MotionPointNet(nn.Module):
    def __init__(
        self,
        pretrain,
        radius,
        nsamples,
        mlps,
        spatial_stride,
        ratio,
        in_channels,
        clip_len,
        num_points,
        drop_out,
        head_type,
        projection_dim,
        device,
        feature_mode: str = "all",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.pretrain = pretrain
        self.radius = radius
        self.nsamples = nsamples
        self.mlps = mlps
        self.strides = spatial_stride
        self.stages = len(self.mlps)
        self.ratio = ratio
        self.in_channels = in_channels
        self.clip_len = clip_len
        self.device = device
        self.feature_mode = feature_mode.lower()

        self.num_points = num_points
        self.shape = [[clip_len, num_points]]
        for s in self.strides:
            self.shape = self.shape + [[clip_len, num_points // s]]
            num_points = self.shape[-1][-1]

        self.in_channels = in_channels
        self.SA_modules = nn.ModuleList()
        for k in range(self.stages):
            # sample times = # stages
            # obtain the in_channels and output channels from the configuration
            channel_list = self.mlps[k].copy()
            channel_list = [in_channels] + channel_list
            channel_out = channel_list[-1]
            # for each sample, may query points multiple times, the query radii and nsamples may be different
            self.SA_modules.append(
                PointNetSAModuleMSG(
                    stride=self.strides[k],
                    radius=self.radius[k],
                    nsamples=self.nsamples[k],
                    length=self.clip_len,
                    channel_list=channel_list,
                )
            )
            in_channels = channel_out
        self.out_channels = channel_out

        if self.pretrain:
            self.latentattn = LatentAttn(
                channel_out, self.clip_len, head=8, num_basis=12, depth=3
            )
            self.loss = nn.CrossEntropyLoss()
        else:
            if head_type == "original":
                self.mlp_head = nn.Sequential(
                    nn.Dropout(drop_out),
                    nn.Linear(channel_out, projection_dim),
                )

            elif head_type == "simclr":
                ### SupCon/SimClr projection head
                self.mlp_head = nn.Sequential(
                    nn.Linear(channel_out, channel_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel_out, projection_dim),
                )
            elif head_type == "linear":
                self.mlp_head = nn.Linear(channel_out, projection_dim)

    def unfold(self, x, shape, mode=1):  # 0: fold 1: unfold
        b, c, _ = x.shape
        if mode == 0:
            x = x.view(b, c, shape[0], shape[1]).permute(0, 2, 1, 3).contiguous()
            x = x.view(b * shape[0], c, shape[1])  # bt, c, n
        if mode == 1:
            x = (
                x.view(b // shape[0], shape[0], c, shape[1])
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            x = x.view(-1, c, shape[0], shape[1])  # b, c, tn
        return x

    def get_input(self, input):
        B, L, N, dim = input.shape
        input = input.view(B * L, -1, dim)
        return (
            input[:, :, :3].contiguous(),
            input[:, :, : self.in_channels].transpose(1, 2).contiguous(),
        )

    def forward_cls_feat(self, xyz, shape, features=None):
        if features is None:
            features = xyz.clone().transpose(1, 2).contiguous()

        output_dict = {"pre_loss": 0}
        for i in range(len(self.SA_modules)):
            xyz, features = self.SA_modules[i](xyz, features, shape)  # B, N, 3; B, C, N
            shape[1] = features.shape[-1]

        if self.pretrain:
            feat = self.unfold(features, shape)  # batch, c, t, n
            xt = feat.max(-1)[0]  # -> batch, c, t
            xn = feat.max(-2)[0]  # -> batch, c, n
            batch, c, n = xn.shape

            xyz = self.unfold(xyz.transpose(1, 2).contiguous(), shape)  # batch, 3, t, n
            xyz_gt = (
                xyz.view(batch, 3, -1).transpose(1, 2).contiguous()
            )  # -> batch, tn, 3

            xn_pre, xyz_pre = self.latentattn(xt, xn)  # batch, c, n  batch, 3, n

            feat_pre = three_interpolation(xyz_gt, xyz_pre, xn_pre).view(
                batch, c, -1, n
            )  # batch, c, t, n
            feat_pre = feat_pre.max(-2)[0].transpose(1, 2).contiguous().view(-1, c)
            feat_pre = F.normalize(feat_pre, dim=-1)  # batch*n, c

            feat_tgt = feat.permute(0, 2, 3, 1).contiguous().view(-1, c)  # batch*tn, c
            feat_tgt = F.normalize(feat_tgt, dim=-1)  # batch*tn, c

            xn = xn.transpose(1, 2).contiguous().view(-1, c)
            xn = F.normalize(xn, dim=-1)  # batch*n, c

            feat_tgt = torch.cat([xn, feat_tgt], dim=0)  # batch*(n+1)

            # InfoNCE loss
            temporal_similarity = (
                torch.matmul(feat_pre, feat_tgt.transpose(0, 1)) / 0.01
            )  # batch*n, batch*t(n+1)
            target_similarity = torch.arange(
                temporal_similarity.shape[0], device=temporal_similarity.device
            )
            loss = self.loss(temporal_similarity, target_similarity)

            output_dict["temporal_similarity"] = temporal_similarity
            output_dict["target_similarity"] = target_similarity
            output_dict["pre_loss"] = loss

        features = self.unfold(features, shape)  # batch, c, t, n
        output_dict["temporal_features"] = features
        output_dict["logit"] = features

        return output_dict

    def forward(self, batch):  # B, L, N, 3
        if batch.ndim == 3:
            batch = batch.unsqueeze(1)
        device = self.device
        input = batch[:, :, :, 0:3]
        B, L, N, _ = input.shape
        shape = [L, N]

        xyz, feats = self.get_input(input)
        output_dict = self.forward_cls_feat(xyz, shape, feats)
        output = None
        if not self.pretrain:
            output = output_dict["temporal_features"]
            output = output.max(2)[0]  # temporal
            output = output.permute(
                0, 2, 1
            )  # strictly only for llava training, we get multiple tokens, so we keep the token dimension and do not max pool spatially
            # print(f"After temporal max pooling, output shape: {output.shape}")  # Debugging line
            # output = output.max(2)[0]  # spatial
            # print(f"After spatial max pooling, output shape: {output.shape}")  # Debugging line
            motion_features = output

        if self.feature_mode == "all":
            #   motion_features = motion_features.unsqueeze(1) unsqueeze for llava.
            return motion_features

        output = self.mlp_head(output)
        # output_dict['logit'] = output
        return output

    def get_trainable_parameters(self):
        """Return list of trainable parameters for optimizer."""
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
