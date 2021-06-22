# -*- coding: utf-8 -*-
# Reference: https://github.com/enriccorona/SMPLicit/blob/main/SMPLicit/network.py
# author: pinakinathc.me

import torch.nn as nn
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, point_pos_size=3, output_dim=1, n_z_cut=12):
        super(Network, self).__init__()
        self.point_pos_size = point_pos_size

        # self.fc0_query = nn.utils.weight_norm(nn.Conv1d(point_pos_size, 128, kernel_size=1, bias=True))
        # self.fc1_query = nn.utils.weight_norm(nn.Conv1d(128, 256, kernel_size=1, bias=True))
        # self.fc2_query = nn.utils.weight_norm(nn.Conv1d(256, 512, kernel_size=1, bias=True))

        self.fc0 = nn.utils.weight_norm(nn.Conv1d(point_pos_size*12 + n_z_cut, 312, kernel_size=1, bias=True))
        self.fc1 = nn.utils.weight_norm(nn.Conv1d(312, 312, kernel_size=1, bias=True))
        self.fc2 = nn.utils.weight_norm(nn.Conv1d(312, 256, kernel_size=1, bias=True))
        self.fc3 = nn.utils.weight_norm(nn.Conv1d(256, 128, kernel_size=1, bias=True))
        self.fc4 = nn.utils.weight_norm(nn.Conv1d(128, output_dim, kernel_size=1, bias=True))

        # self.fc0 = nn.utils.weight_norm(nn.Linear(512 + n_z_cut, 312))
        # self.fc1 = nn.utils.weight_norm(nn.Linear(312, 312))
        # self.fc2 = nn.utils.weight_norm(nn.Linear(312, 256))
        # self.fc3 = nn.utils.weight_norm(nn.Linear(256, 128))
        # self.fc4 = nn.utils.weight_norm(nn.Linear(128, output_dim))        

        self.activation = F.relu
        #self.activation = torch.sin

    def forward(self, uvmap_representation, point_position):
        _B = len(uvmap_representation)
        _numpoints = len(point_position[0])

        point_encoding = point_position
        point_encoding = point_encoding.reshape(_B, _numpoints, self.point_pos_size).permute(0,2,1)

        # x_position = self.activation(self.fc0_query(point_encoding))
        # x_position = self.activation(self.fc1_query(x_position))
        # x_position = self.activation(self.fc2_query(x_position))

        ## Positional Encoding of coordinated
        x_position = []
        for f in [1, 2, 4, 8, 16, 32]:
            x_position.append(torch.sin(f*point_encoding))
            x_position.append(torch.cos(f*point_encoding))
        x_position = torch.cat(x_position, 1) # Shape: Nx36

        uvmap_representation = uvmap_representation.unsqueeze(-1).repeat(1, 1, _numpoints)
        _in = torch.cat((x_position, uvmap_representation), 1)
        # _in = _in.permute(0, 2, 1) # B x 5000 x dim

        x = self.fc0(_in)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        # x = self.activation(x)
        # x = x.permute(0, 2, 1) # B x 1 x 5000

        if x.shape[1] == 1:
            return x[:, 0]
        else:
            return x


if __name__ == '__main__':
    print ('Testing data flow of Generator network')
    net = Network(point_pos_size=3, output_dim=1, n_z_cut=360)
    _B = 3
    z_cut = torch.randn(_B, 360)
    point = torch.randn(_B, 5000, 3)

    out = net(z_cut, point)
    print ('out shape: ', out.shape)
