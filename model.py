# -*- coding: utf-8 -*-
# author: pinakinathc

import torch
import torch.nn as nn
import torchvision
from networks.encoder import HGFilter as F
from networks.generator import Network as G

class MultiViewSk(nn.Module):
	def __init__(self, opt):
		super(MultiViewSk, self).__init__()
		self.opt = opt

		# self.encoder = F(self.opt.num_stack, self.opt.norm, self.opt.hg_down, self.opt.num_hourglass, self.opt.hourglass_dim)
		self.encoder = nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-1])
		self.transform_uvmap_func = nn.Sequential(
			nn.Linear(2048, opt.n_z_cut),
			nn.ReLU())
		self.generator = G(point_pos_size=self.opt.point_pos_size, n_z_cut=self.opt.n_z_cut)

	def forward(self, uvmap, points):
		_B = uvmap.shape[0]

		## Old HourGlass code
		# uvmap_representation = self.encoder(uvmap)[0][-1] # not training intermediate representation as in PIFu
		# uvmap_rep = uvmap_representation.reshape(_B, self.opt.hourglass_dim, -1).mean(dim=-1)
		# latent_vec = self.transform_uvmap_func(uvmap_rep.view(-1, self.opt.num_views, uvmap_rep.shape[1]).mean(dim=1))

		## New ResNet code
		uvmap_representation = self.encoder(uvmap).reshape(_B, -1)
		assert list(uvmap_representation.shape) == [_B, 2048]
		latent_vec = self.transform_uvmap_func(uvmap_representation)
		pred = self.generator(latent_vec, points) # generator handles encoding of points
		return pred, latent_vec


if __name__ == '__main__':
	print ('Testing data flow of MultiViewSk')
	from options import BaseOptions
	# get options/parameters
	opt = BaseOptions().parse()

	net = MultiViewSk(opt)
	_B = 3
	inp = torch.randn(_B * opt.num_views, 3, 256, 256)
	points = torch.randn(_B, 5000, 3)
	pred, latent_vec = net(inp, points)
	print ('Shape of predictions: {}; latent vector: {}'.format(pred.shape, latent_vec.shape))
