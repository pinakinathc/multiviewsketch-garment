# -*- coding: utf-8 -*-

import torch
import numpy as np

import os
import datetime
import torch
import random
import glob
from torch.autograd import Variable
import torchvision.models as models
import torch.optim as optim
import sys
import time
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size=512,
        dims=[ 512, 512, 512, 512, 512, 512, 512, 512 ],
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


class Updater(nn.Module):
    def __init__(self):
        super(Updater, self).__init__()

        kdim = 512

        self.layer = nn.Sequential(
            nn.Linear(2*kdim+10, kdim),
            nn.ReLU())

    def forward(self, latent_vec, feat, pos_emb):
        ''' Shape of latent_vec: B x 512; feat: B x 512 '''
        comb_feat = torch.cat([latent_vec, feat, pos_emb], dim=-1)
        return self.layer(comb_feat)


class AlignNet(nn.Module):
    def __init__(self, img_feat_dim=512, azi_feat_dim=10):
        super(AlignNet, self).__init__()

        kdim = 512

        self.layer = nn.Sequential(
            nn.Linear(img_feat_dim+azi_feat_dim, kdim),
            nn.ReLU())

    def forward(self, comb_feat):
        # comb_feat = torch.cat([img_vec, pos_vec])
        return self.layer(comb_feat)


class AlignUpdater(nn.Module):
    def __init__(self, img_feat_dim=512, azi_feat_dim=10):
        super(AlignUpdater, self).__init__()

        self.kdim = 1024

        self.layer = nn.Sequential(
            nn.Linear(img_feat_dim+azi_feat_dim, self.kdim//2),
            nn.BatchNorm1d(self.kdim//2),
            nn.ReLU(),
            nn.Linear(self.kdim//2, self.kdim),
            nn.BatchNorm1d(self.kdim),
            nn.ReLU()
        )

        self.feat_emb = nn.Sequential(
            nn.Linear(self.kdim//2, self.kdim//2),
            nn.BatchNorm1d(self.kdim//2),
            nn.ReLU(),
            nn.Linear(self.kdim//2, self.kdim//2),
            # nn.BatchNorm1d(self.kdim//2),
            # nn.ReLU()
        )

        self.alpha_emb = nn.Sequential(
            nn.Linear(self.kdim//2, self.kdim//2),
            nn.BatchNorm1d(self.kdim//2),
            nn.ReLU(),
            nn.Linear(self.kdim//2, self.kdim//2),
            # nn.BatchNorm1d(self.kdim//2),
            # nn.ReLU()
        )

    def forward(self, comb_feat):
        output_feat = self.layer(comb_feat) # B x 1024
        feat = self.feat_emb(output_feat[:, :self.kdim//2])
        alpha = self.alpha_emb(output_feat[:, self.kdim//2:])
        return feat, alpha


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(
            *(list(self.model.children())[:-1]),
            nn.Flatten()
        )
    
    def forward(self, inp):
        return self.model(inp)


class AlphaClassifier(nn.Module):
    """ Tries to classify alpha into 0-360 azimuth """
    def __init__(self, feat_dim=512):
        super(AlphaClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.BatchNorm1d(feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, 36),
        )
    
    def forward(self, alpha):
        return self.classifier(alpha)
