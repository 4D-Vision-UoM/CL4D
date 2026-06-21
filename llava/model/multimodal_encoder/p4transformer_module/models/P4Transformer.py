import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "modules"))

from point_4d_convolution import *
from transformer import *


class P4Transformer(nn.Module):
    def __init__(
        self,
        radius,
        nsamples,
        spatial_stride,  # P4DConv: spatial
        temporal_kernel_size,
        temporal_stride,  # P4DConv: temporal
        emb_relu,  # embedding: relu
        dim,
        depth,
        heads,
        dim_head,  # transformer
        mlp_dim,
        head_type,
        projection_dim,
        device,
        feature_mode="all",
    ):  # output
        super().__init__()

        self.tube_embedding = P4DConv(
            in_planes=0,
            mlp_planes=[dim],
            mlp_batch_norm=[False],
            mlp_activation=[False],
            spatial_kernel_size=[radius, nsamples],
            spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            temporal_padding=[1, 0],
            operator="+",
            spatial_pooling="max",
            temporal_pooling="max",
        )

        self.pos_embedding = nn.Conv1d(
            in_channels=4,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.emb_relu = nn.ReLU() if emb_relu else False
        self.device = device
        self.feature_mode = feature_mode

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        if head_type == "linear":
            self.mlp_head = nn.Linear(dim, projection_dim)
        elif head_type == "simclr":
            ### SupCon/SimClr projection head
            self.mlp_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, projection_dim),
            )
        elif head_type == "original":
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, projection_dim),
            )

    def forward(self, batch):  # [B, L, N, 3]
        device = self.device
        input = batch[:, :, :, 0:3]  # Extract only xyz (first 3 channels)
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones(
                (xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device
            ) * (t + 1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(
            input=xyzts,
            shape=(xyzts.shape[0], xyzts.shape[1] * xyzts.shape[2], xyzts.shape[3]),
        )  # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)  # [B, L,   n, C]
        features = torch.reshape(
            input=features,
            shape=(
                features.shape[0],
                features.shape[1] * features.shape[2],
                features.shape[3],
            ),
        )  # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)

        if self.feature_mode == "all":
            return output

        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output

    def get_trainable_parameters(self):
        """Return list of trainable parameters for optimizer."""
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
