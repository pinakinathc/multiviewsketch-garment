# -*- coding: utf-8 -*-
# author: pinakinathc

import os
import time
import torch
from dataloader import GarmentDataset
from utils import projection, adjust_learning_rate
from model import MultiViewSk
from options import BaseOptions
import numpy as np
from torch.utils.data import DataLoader
from utils import visualise_NDF
from sdf import create_grid

# get options
opt = BaseOptions().parse()

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d'%opt.gpu_id)

    train_dataset = GarmentDataset(opt, phase='test')
    test_dataset = GarmentDataset(opt, phase='test')

    # create data loader
    train_data_loader = DataLoader(train_dataset,
        batch_size=1, shuffle=not opt.serial_batches,
        num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print ('train data size: ', len(train_data_loader))
    
    # create net
    device_ids = list(map(int, opt.gpu_ids.split(',')))
    model = MultiViewSk(opt)
    if len(device_ids) > 1:
        print ('using multi-gpu')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device=cuda)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate)
    lr = opt.learning_rate
    criterion = torch.nn.L1Loss()

    # load checkpoints
    if opt.load_checkpoint_path is not None:
        print ('loading model weights from: %s'% opt.load_checkpoint_path)
        model.load_state_dict(torch.load(opt.load_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        model_path = '%s/%s/model_latest'% (opt.checkpoints_path, opt.name)
        print ('resuming from: %s'% model_path)
        model.load_state_dict(torch.load(model_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s'% (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s'% (opt.results_path, opt.name), exist_ok=True)

    # testing
    model.eval()
    for train_idx, train_data in enumerate(train_data_loader):
        iter_start_time = time.time()

        # retrieve the data
        image_tensor = train_data['img'].to(device=cuda)
        calib_tensor = train_data['calib'].to(device=cuda)
        ## sample_tensor = train_data['samples'].to(device=cuda)
        label_tensor = train_data['labels'].to(device=cuda)
        
        # reshape multiview tensor
        image_tensor = image_tensor.view(
            image_tensor.shape[0] * image_tensor.shape[1],
            image_tensor.shape[2], image_tensor.shape[3], image_tensor.shape[4])

        b_min = train_data['b_min'].detach().cpu().numpy()[0]
        b_max = train_data['b_max'].detach().cpu().numpy()[0]
        print (b_min, b_max)
        samples, calibs = create_grid(opt.resolution, opt.resolution, opt.resolution, b_min-0.1, b_max+0.1)
        assert list(samples.shape) == [3, opt.resolution, opt.resolution, opt.resolution], 'unexpected shape: {}'.format(samples.shape)
        samples = samples.reshape(3, -1).T # Nx3
        sample_tensor = torch.Tensor(samples).float().unsqueeze(0).to(device=cuda) # batch size 1

        pred, latent_vec = model(image_tensor, sample_tensor)

        visualise_NDF(pred.detach().cpu().numpy().reshape(opt.resolution, opt.resolution, opt.resolution))
        from skimage import measure
        from utils import save_obj_mesh
        verts, faces, normals, values = measure.marching_cubes_lewiner(pred.detach().cpu().numpy().reshape(opt.resolution, opt.resolution, opt.resolution), 0.02)
        verts = (np.matmul(calibs[:3, :3], verts.T) + calibs[:3, 3:4]).T
        save_path = os.path.join(os.getcwd(), opt.results_path, opt.name, train_data['name'][0]+'_pred.obj')
        save_obj_mesh(save_path, verts, faces)


if __name__ == '__main__':
    train(opt)          
