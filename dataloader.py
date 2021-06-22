# -*- coding: utf-8 -*-
# Reference: https://github.com/shunsukesaito/PIFu/blob/master/lib/data/TrainDataset.py
# author: pinakinathc.me

from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import math
import igl # calculates signed distance field
# import kaolin # calculates SDF
from utils import make_rotate, visualise_NDF
from sdf import create_grid

def load_trimesh(root_dir):
    # folders = os.listdir(root_dir)
    folders = ['A1YENASWUJWB'] ## Overfit on 1 shirt
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        mesh = trimesh.load(os.path.join(root_dir, f, 'shirt_mesh_r_tmp.obj'))
        ## Rescale
        scene_obj = trimesh.Scene(mesh)
        scene_obj = scene_obj.scaled(0.9/scene_obj.scale)
        mesh = scene_obj.geometry['shirt_mesh_r_tmp.obj']
        mesh.rezero()
        meshs[sub_name] = mesh
    return meshs


def save_samples_truncted_prob(fname, points, prob):
    ''' Save the visualization of sampling to a ply file '''
    red = prob.reshape([-1, 1])/prob.max() * 255
    green = red
    blue = np.zeros(red.shape)

    to_save = np.concatenate([points, red, green, blue], axis=-1)
    return np.savetxt(fname, to_save, fmt='%.6f %.6f %.6f %d %d %d', comments='',
        header=('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\
            \nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(points.shape[0]))


class GarmentDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.render = os.path.join(self.root, 'RENDER')
        self.obj = os.path.join(self.root, 'GEO', 'OBJ')

        self.is_train = (phase == 'train')
        self.load_size = self.opt.load_size # load size of input image
        self.num_views = self.opt.num_views
        self.num_sample_inout = self.opt.num_sample_inout # no. of sampling points


        # self.yaw_list = list(range(0, 360, 1))
        self.yaw_list = [0] ## Overfit on 1 view
        self.pitch_list = [0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # augmentation, not much relevant for binary sketches
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con,
                saturation=opt.aug_sat, hue=opt.aug_hue)])

        self.mesh_dic = load_trimesh(self.obj)

    def get_subjects(self):
        # all_subjects = os.listdir(self.render)
        all_subjects = ['A1YENASWUJWB']
        # all_subjects = ['A1YENASWUJWB', 'A2UOBZJIFYZI']
        val_subjects = np.loadtxt('val.txt', dtype=str)

        if len(val_subjects) == 0:
            return all_subjects
        if self.is_train:
            return sorted(list(set(all_subjects) - set(val_subjects)))
        else:
            return sorted(list(val_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''return: 'img': [num_views, C, W, H] images '''
        pitch = self.pitch_list[pid]

        # view ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]

        local_state = np.random.RandomState()
        if random_sample and self.is_train:
            view_ids = local_state.choice(self.yaw_list, num_views, replace=False)

        render_list = []

        for idx, vid in enumerate(view_ids):
            vid = (vid+180) % 360
            render_path = os.path.join(self.render, subject, '%d_%d_%02d.png'%(vid, pitch, 0))

            render = Image.open(render_path).convert('RGBA').split()[-1].convert('RGB') # sketch is in alpha channel

            if self.is_train:
                # pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random scale
                if self.opt.random_scale and np.random.randn() > 0.5:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)), int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)), int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            render = self.to_tensor(render)
            render_list.append(render)

        return {
            'img': torch.stack(render_list, dim=0),
        }

    ## Sampling uniformly across the grid
    def select_sampling_method(self, subject):
        mesh = self.mesh_dic[subject]
        vertices = mesh.vertices
        b_max = vertices.max(0)
        b_min = vertices.min(0)
        delta = 0.1

        resolution = self.opt.resolution
        samples, calibs = create_grid(resolution, resolution, resolution, b_min-delta, b_max+delta)
        assert list(samples.shape) == [3, resolution, resolution, resolution], 'unexpected shape: {}'.format(samples.shape)
        samples = samples.reshape(3, -1).T # Nx3

        # Selecting random points
        if self.is_train:
            local_state = np.random.RandomState()
            idx = local_state.choice(samples.shape[0], 10000) # Number of samples
            samples = samples[idx]

        labels = np.abs(igl.signed_distance(samples, mesh.vertices, mesh.faces)[0]) # Calculate UDF
        # smpl_mesh = kaolin.rep.TriangleMesh.from_tensors(torch.Tensor(mesh.vertices.astype(np.float64)).cuda(), torch.Tensor(mesh.faces, dtype=torch.long).cuda())
        # smpl_mesh_sdf = kaolin.conversions.trianglemesh_to_sdf(smpl_mesh)
        # labels = smpl_mesh_sdf(torch.Tensor(samples).cuda())

        # visualise_NDF(labels.reshape(resolution, resolution, resolution))
        if not self.is_train:
            from skimage import measure
            from utils import save_obj_mesh
            verts, faces, normals, values = measure.marching_cubes_lewiner(labels.reshape(resolution, resolution, resolution), 0.01)
            verts = np.matmul(calibs[:3, :3], verts.T) + calibs[:3, 3:4]
            save_path = os.path.join(os.getcwd(), self.opt.results_path, self.opt.name, subject+'_gt')
            save_obj_mesh(save_path+'.obj', verts.T, faces)
            save_samples_truncted_prob(save_path+'.ply', samples, labels)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        calibs = torch.Tensor(calibs).float()

        del mesh

        return {
            'samples': samples,
            'labels': labels,
            'calib': calibs,
            'b_min': torch.tensor(b_min),
            'b_max': torch.tensor(b_max)
        }


    def get_item(self, index):
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.obj, subject+'.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid
        }
        render_data = self.get_render(subject, num_views=self.num_views,
            yid=yid, pid=pid, random_sample=self.opt.random_multiview)

        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)

        return res

    def __getitem__(self, index):
        return self.get_item(index)


if __name__ == '__main__':
    print ('Testing GarmentDataset')
    from torch.utils.data import DataLoader
    from options import BaseOptions

    opt = BaseOptions().parse()
    train_dataset = GarmentDataset(opt, phase='train')
    test_dataset = GarmentDataset(opt, phase='test')

    # create data loader
    train_data_loader = DataLoader(train_dataset,
        batch_size=opt.batch_size, shuffle=not opt.serial_batches,
        num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print ('train data size: ', len(train_data_loader))

    # Note: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
        batch_size=1, shuffle=False, num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print ('test data size: ', len(test_data_loader))

    for train_idx, train_data in enumerate(train_data_loader):
        # retrieve the data
        image_tensor = train_data['img']
        sample_tensor = train_data['samples']
        calib_tensor = train_data['calib']
        label_tensor = train_data['labels']

        print ('shape of image_tensor: {}, calib_tensor: {}, sample_tensor: {}, label_tenspr: {}'.format(
            image_tensor.shape, calib_tensor.shape, sample_tensor.shape, label_tensor.shape))
