# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss.chamfer import chamfer_distance
from igl import signed_distance
from skimage import measure


def save_vertices_ply(fname, points, prob=None):
	'''
	Save the visualization of sampling to a ply file.
	Red points represent positive predictions.
	Green points represent negative predictions.
	:param fname: File name to save
	:param points: [N, 3] array of points
	:param prob: [N, 1] array of predictions in the range [0~1]
	:return:
	'''
	if prob is None:
		prob = np.ones((points.shape[0], 1))
	r = (prob > 0).reshape([-1, 1]) * 255
	g = (prob < 0).reshape([-1, 1]) * 255
	b = np.zeros(r.shape)

	to_save = np.concatenate([points, r, g, b], axis=-1)
	print ('writing to: ', fname)
	return np.savetxt(fname,
	  to_save,
	  fmt='%.6f %.6f %.6f %d %d %d',
	  comments='',
	  header=(
		  'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
		  points.shape[0]))


def svg2img(render_path, p_mask=None):
	""" Reads SVG image and converts to render partial sketch """
	doc = minidom.parse(render_path)
	height = int(doc.getElementByTagName("svg")[0].getAttribute("height"))
	width = int(doc.getElementByTagName("svg")[0].getAttribute("width"))
	raster_image = np.zeros((height, width), dtype=np.float32)
	path_strings = [path.getAttribute("d") for path in doc.getElementByTagName("path")]
	doc.unlink()
	Len = len(path_strings)
	for path_str in path_strings:
		if np.random.randn() <= p_mask:
			continue
		path = parse_path(path_str)
		for e in path:
			if isinstance(e, Line):
				x0 = round(e.start.real)
				y0 = round(e.start.imag)
				x1 = round(e.end.real)
				y1 = round(e.end.imag)
				cordList = list(bresenham(x0, y0, x1, y1))
				for cord in cordList:
					raster_image[cord[1], cord[0]] = 255.0
	raster_image = 255.0 - scipy.ndimage.binary_dilation(raster_image) * 255.0
	return Image.fromarray(raster_image).convert("RGB")


def calculate_npr_dist(filename1, filename2):
	npr1 = np.array(Image.open(filename1).convert('RGBA').split()[-1].convert('RGB'))[:,:,0]
	npr2 = np.array(Image.open(filename2).convert('RGBA').split()[-1].convert('RGB'))[:,:,0]

	points_npr1 = torch.tensor(np.array(
		np.where(npr1 > 0)[:2]).reshape(2, -1).T, dtype=torch.float32).unsqueeze(0)
	points_npr2 = torch.tensor(np.array(
		np.where(npr2 > 0)[:2]).reshape(2, -1).T, dtype=torch.float32).unsqueeze(0)

	# return distance_fn(points_npr1, points_npr2, bidirectional=True)
	return chamfer_distance(points_npr1, points_npr2)[0]


def calculate_mesh_view_dist(mesh1, mesh2, view):
	view = (view+180)%360
	reso = 256
	grid = np.mgrid[-.9:.9:reso*1j, -.9:.9:reso*1j, -.9:.9:reso*1j].reshape(3, -1).T
	
	# resize mesh
	def get_subverts(mesh):
		scene = trimesh.Scene(mesh)
		scene = scene.scaled(1.5/scene.scale)
		key = list(scene.geometry.keys())[0]
		mesh = scene.geometry[key]
		mesh.rezero()
		mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
		sdf = signed_distance(grid, mesh.vertices, mesh.faces)[0][:, np.newaxis]
		sdf = sdf.reshape((reso, reso, reso))
		verts, faces, normals, values = measure.marching_cubes(sdf, 0.0)
		ray_direction = np.array(
			[[np.sin(np.radians(view)), 0, np.cos(np.radians(view))]]).T
		dot_product = np.dot(normals, ray_direction)[:, 0]
		idx = np.where(dot_product >= 0.0)
		subset_verts = verts[idx]
		return subset_verts

	points1 = np.array(get_subverts(mesh1))
	points2 = np.array(get_subverts(mesh2))
	N = min(points1.shape[0], points2.shape[0])
	points1 = points1[np.random.choice(np.arange(points1.shape[0]), N, replace=True)]
	points2 = points2[np.random.choice(np.arange(points2.shape[0]), N, replace=True)]
	points1 = torch.tensor(points1, dtype=torch.float32).unsqueeze(0)
	points2 = torch.tensor(points2, dtype=torch.float32).unsqueeze(0)
	return chamfer_distance(points1, points2)[0]/N
