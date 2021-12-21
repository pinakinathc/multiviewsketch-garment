import time
import os
import glob
import tqdm
import numpy as np
import trimesh
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from igl import signed_distance

# Image transforms
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

class GarmentDataset(torch.utils.data.Dataset):
    def __init__(self,
            data_dir,
            val_path,
            num_views=3,
            num_points=2048,
            use_partial=False,
            resolution=256,
            evaluate=False):
        
        self.data_dir = data_dir
        self.num_views = num_views
        self.num_points = num_points
        self.use_partial = use_partial
        self.reso = resolution
        self.evaluate = evaluate

        self.all_garments = os.listdir(os.path.join(
            self.data_dir, 'GEO', 'OBJ'))
        val_garments = np.loadtxt(val_path, dtype=str)

        print ('Total number of garments found: %d'%len(self.all_garments))
        if not self.evaluate:
            print ('Training set')
            self.all_garments = list(set(self.all_garments) - set(val_garments))[:10]
        else:
            print ('Testing set')
            self.all_garments = list(val_garments)
        # self.all_garments = ['watertight_100']
        # self.all_garments = ['W8EXCGDKFZST']
        print ('Number of garments used: %d'%len(self.all_garments))

        self.mesh_list = {}
        self.mesh_centroid = {}
        self.mesh_scale = {}
        self.azi_list = list(range(0, 360, 10))

        tmp_list_garments = []
        print ('Starting to read all meshes. Please wait...')
        start_time = time.time()
        for garment in tqdm.tqdm(self.all_garments):
            mesh_path = glob.glob(os.path.join(
                self.data_dir, 'GEO', 'OBJ', garment, '*.obj'))[0]
            mesh = trimesh.load(mesh_path)

            self.mesh_centroid[garment] = mesh.centroid
            mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
            scene = trimesh.Scene(mesh)
            self.mesh_scale[garment] = scene.scale
            scene = scene.scaled(1.5/scene.scale)
            mesh = scene.geometry[list(scene.geometry.keys())[0]]

            # scene = trimesh.Scene(mesh)
            # scene = scene.scaled(1.5/scene.scale)
            # mesh = scene.geometry[list(scene.geometry.keys())[0]]
            # mesh.rezero()
            # mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)


            self.mesh_list[garment] = mesh
            tmp_list_garments.append(garment)
        print ('Time taken to read all meshe: ', time.time() - start_time)
        self.all_garments = tmp_list_garments

    def __len__(self):
        if self.evaluate:
            return len(self.all_garments)
        else:
            return int(1e7)

    def __getitem__(self, index):
        index = index % len(self.all_garments)
        local_state = np.random.RandomState()

        garment = self.all_garments[index]
        mesh = self.mesh_list[garment]

        all_img_tensor = [] # Store image from input sketch
        all_azi_tensor = []
        all_pos_emb_feat = [] # Store view or positional encoding
        all_xyz = [] # Store xyz points
        all_sdf = [] # Store SDF values

        # Randomly select n views
        azi_list = local_state.choice(self.azi_list, self.num_views).tolist()

        for azi in azi_list:

            # Read image sketch for a view
            img_path = os.path.join(self.data_dir, 'RENDER', garment, '%d_0_00.png'%azi)
            img = Image.open(img_path).convert('RGBA').split()[-1].convert('RGB')
            img_tensor = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
            all_img_tensor.append(img_tensor)

            # Compute azi tensor
            all_azi_tensor.append(torch.LongTensor([azi//10]))

            # Compute positional encoding
            pos_emb_feat = []
            for p in [1, 2, 4, 8, 16]:
                pos_emb_feat.append(np.sin(np.radians(azi*p)))
                pos_emb_feat.append(np.cos(np.radians(azi*p)))
            all_pos_emb_feat.append(pos_emb_feat)

            """ Random Sampling """
            if not self.use_partial:
                surface_xyz, _ = trimesh.sample.sample_surface(mesh, 3*self.num_points//4)
            else:
                partial_mesh = trimesh.load(os.path.join(
                    self.data_dir, 'partial_view_ply', garment, '%d.ply'%azi
                ))
                surface_xyz = self.resize_ply(
                    partial_mesh, self.mesh_scale[garment], self.mesh_centroid[garment])
                surface_xyz = surface_xyz[local_state.choice(
                    np.arange(surface_xyz.shape[0]), 3*self.num_points//4)]
            near_xyz = surface_xyz + local_state.normal(scale=0.03, size=surface_xyz.shape)

            B_MAX = near_xyz.max(0)
            B_MIN = near_xyz.min(0)
            length = (B_MAX - B_MIN)
            far_xyz = local_state.rand(self.num_points//4, 3) * length + B_MIN
            xyz = np.concatenate([near_xyz, far_xyz], 0)
            local_state.shuffle(xyz)

            # Calculate SDF of sampled points
            sdf = signed_distance(xyz, mesh.vertices, mesh.faces)[0][:, np.newaxis]

            # Sample random points
            all_xyz.append(xyz)

            # Compute SDF
            all_sdf.append(sdf)

        all_img_tensor = torch.cat(all_img_tensor, dim=0)
        all_azi_tensor = torch.cat(all_azi_tensor, dim=0)
        all_pos_emb_feat = Variable(torch.FloatTensor(all_pos_emb_feat))
        all_xyz = Variable(torch.FloatTensor(all_xyz))
        all_sdf = Variable(torch.FloatTensor(all_sdf))
        
        return (
            all_img_tensor,
            all_pos_emb_feat,
            all_xyz,
            all_sdf,
            all_azi_tensor
        )

    def resize_ply(self, mesh, scale, centroid):
        mesh = trimesh.Trimesh(mesh.vertices - centroid)
        scene = trimesh.Scene(mesh)
        scene = scene.scaled(1.5/scale)
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
        return mesh.vertices


if __name__ == '__main__':
    from options import opts
    from utils import save_vertices_ply

    dataset = GarmentDataset(
        data_dir=opts.data_dir,
        val_path=os.path.join(opts.data_dir, 'val.txt'),
        num_views=opts.num_views,
        num_points=opts.num_points,
        # use_partial=True
    )

    count = 0
    for all_img, all_pos_emb_feat, all_xyz, all_sdf, all_azi_tensor in dataset:
        for idx in range(opts.num_views):
            print ('shape of points: {}, sdf: {}'.format(
                all_xyz[idx].shape, all_sdf[idx].shape
            ))
            save_vertices_ply(
                os.path.join('output', '%d.ply'%count),
                all_xyz[idx], all_sdf[idx]
            )
            count += 1
            input ('check')