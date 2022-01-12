# -*- coding: utf-8 -*-

import argparse
import trimesh
import torch
import torchvision
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights,
    diffuse, 
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)

parser = argparse.ArgumentParser(description="Visualise obj mesh")
parser.add_argument('--obj_path', type=str, default='./', help='enter path to obj mesh')
parser.add_argument('--output_path', type=str, default='./', help='enter path to rendered images')
opt = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

mesh = trimesh.load(opt.obj_path)
mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
scene = trimesh.Scene(mesh)
scene = scene.scaled(3/scene.scale)
mesh = scene.geometry[list(scene.geometry.keys())[0]]
verts = torch.tensor(mesh.vertices, dtype=torch.float)
faces = torch.tensor(mesh.faces, dtype=torch.long)
# verts_rgb = torch.tensor(mesh.visual.vertex_colors[:, :3], dtype=torch.float)[None]
# verts_rgb = verts_rgb/256.0

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object
mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)],
    textures=textures
)

batch_size = 1

# Create Renderer
meshes = mesh.extend(batch_size)
azim = torch.linspace(-180, 180, batch_size)
azim = torch.tensor([180])
R, T = look_at_view_transform(dist=2.7, elev=0, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
# lights = AmbientLights(device=device)
# lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

images = renderer(meshes, cameras=cameras, lights=lights)
torchvision.utils.save_image(images.permute(0, 3, 1, 2)[:, :3, :, :], opt.output_path)