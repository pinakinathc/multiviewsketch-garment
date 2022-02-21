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
            self.all_garments = list(set(self.all_garments) - set(val_garments))
        else:
            print ('Testing set')
            self.all_garments = list(val_garments)
        print ('Number of garments used: %d'%len(self.all_garments))

        self.azi_list = list(range(0, 360, 10))
        
        self.points_data = {}
        if not self.use_partial:
            for garment in tqdm.tqdm(self.all_garments):
                self.points_data[garment] = np.load(os.path.join(
                    self.data_dir, 'all_mesh_points', '%s.npy'%garment), allow_pickle=True)

    def __len__(self):
        if self.evaluate:
            return len(self.all_garments)
        else:
            return int(1e7)

    def __getitem__(self, index):
        index = index % len(self.all_garments)
        local_state = np.random.RandomState()

        garment = self.all_garments[index]

        all_img_tensor = [] # Store image from input sketch
        all_azi_tensor = []
        all_pos_emb_feat = [] # Store view or positional encoding
        all_xyz = [] # Store xyz points
        all_sdf = [] # Store SDF values
        all_mask = [] # Store Mask values to simulate random views

        # Randomly select n views
        # azi_list = local_state.choice(self.azi_list, self.num_views).tolist()
        azi_list = local_state.choice(self.azi_list, 1).tolist()
        delta_azi = (36 // self.num_views) * 10
        for i in range(self.num_views-1):
            azi_list.append((azi_list[-1] + delta_azi)%360)

        views = local_state.randint(1, self.num_views)

        for id_azi, azi in enumerate(azi_list):

            """ Random Sampling """
            data_list = []
            key_list = []
            if not self.use_partial:
                tmp_garment = garment
                # points_data = np.load(os.path.join(
                #     self.data_dir, 'all_mesh_points', '%s.npy'%garment), allow_pickle=True)
                points_data = self.points_data[garment]
                key_list = ['inside', 'outside', 'random']
            else:
                if local_state.uniform() > 0.7:
                    closest_shirts = np.loadtxt(os.path.join(
                        self.data_dir, 'closest_mesh', '%s.txt'%garment
                    ), dtype=str).tolist()
                    closest_shirts = list(set(closest_shirts) & set(self.all_garments))
                    closest_shirts = closest_shirts[:10]
                    prob_weights = 3**np.arange(len(closest_shirts), 0, -1)
                    prob_weights = prob_weights / prob_weights.sum()
                    tmp_garment = local_state.choice(closest_shirts, 1, p=prob_weights)[0]
                    key_list = ['inside', 'outside', 'inside']
                else:
                    tmp_garment = garment
                    key_list = ['inside', 'outside', 'random']
                points_data = np.load(os.path.join(
                    self.data_dir, 'partial_mesh_points', tmp_garment, '%d.npy'%azi
                ), allow_pickle=True)
            for keys in key_list:
                data = points_data.item().get(keys)
                data_list.append(data[local_state.choice(data.shape[0], self.num_points//3)])
            data = np.concatenate(data_list, 0)
            local_state.shuffle(data)
            xyz = data[:, :3]
            sdf = data[:, 3][:, np.newaxis]

            if id_azi >= views:
                mask = np.zeros_like(sdf)
            else:
                mask = np.ones_like(sdf)

            all_xyz.append(xyz)
            all_sdf.append(sdf)
            all_mask.append(mask)

            # Read image sketch for a view
            img_path = os.path.join(self.data_dir, 'RENDER', tmp_garment, '%d_0_00.png'%azi)
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

        all_img_tensor = torch.cat(all_img_tensor, dim=0)
        all_azi_tensor = torch.cat(all_azi_tensor, dim=0)
        all_pos_emb_feat = Variable(torch.FloatTensor(all_pos_emb_feat))
        all_xyz = Variable(torch.FloatTensor(np.array(all_xyz)))
        all_sdf = Variable(torch.FloatTensor(np.array(all_sdf)))
        all_mask = Variable(torch.FloatTensor(np.array(all_mask)))
        
        return (
            all_img_tensor,
            all_pos_emb_feat,
            all_xyz,
            all_sdf,
            all_mask,
            all_azi_tensor
        )


if __name__ == '__main__':
    from options import opts
    from utils import save_vertices_ply
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader

    dataset = GarmentDataset(
        data_dir=opts.data_dir,
        val_path=os.path.join(opts.data_dir, 'val.txt'),
        num_views=opts.num_views,
        num_points=opts.num_points,
        use_partial=False,
        evaluate=True
    )

    data_loader = DataLoader(
        dataset=dataset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers
    )

    count = 0
    for all_img, all_pos_emb_feat, all_xyz, all_sdf, all_mask, all_azi in tqdm.tqdm(data_loader):
        print ('Shape: all_img {}, all_pos_emb_feat {}, all_xyz {}, all_sdf {} all_mask {} all_azi {}'.format(
            all_img.shape, all_pos_emb_feat.shape, all_xyz.shape, all_sdf.shape, all_mask.shape, all_azi.shape
        ))
        # continue
        for idx in range(opts.num_views):
            print ('shape of points: {}, sdf: {}'.format(
                all_xyz[idx].shape, all_sdf[idx].shape))
            print ('max: {}, min: {}'.format(all_sdf[idx].max(), all_sdf[idx].min()))
            save_vertices_ply(
                os.path.join('output', '%d.ply'%count),
                all_xyz[0][idx], all_sdf[0][idx])
            save_image(all_img[idx], os.path.join('output', '%d.jpg'%count))
            count += 1
            input ('check')