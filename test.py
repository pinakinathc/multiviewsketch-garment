# -*- coding: utf-8 -*-
# author: pinakinathc.me

import os
import numpy as np
from tqdm import tqdm
import random
random.seed()
import torch
from dataloader import GarmentDataset
from torch.utils.data import DataLoader
from options import BaseOptions
from model import MultiViewSk
from utils import gen_mesh


# get options
opt = BaseOptions().parse()

def test(opt):
    # set cuda
    cuda = torch.device('cuda:%d'% opt.gpu_id)

    test_dataset = GarmentDataset(opt, phase='test')
    # create data loader
    # batch size should be 1
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
        num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print ('test data size: ', len(test_data_loader))

    # create net
    device_ids = list(map(int, opt.gpu_ids.split(',')))
    model = MultiViewSk(opt)
    if len(device_ids) > 1:
        print ('using multi-gpu')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device=cuda)

    # load checkpoints
    if opt.load_checkpoint_path is not None:
        print ('loading model from: ', opt.load_checkpoint_path)
        model.load_state_dict(torch.load(opt.load_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/model_latest'% (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/model_epoch_%d'% (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print ('resuming from: ', model_path)
        model.load_state_dict(torch.load(model_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s'% (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s'% (opt.results_path, opt.name), exist_ok=True)

    epoch = 0
    with torch.no_grad():
        model.eval()
        for idx, test_data in enumerate(test_dataset):
            save_path = '%s/%s/test_eval_epoch%d_%s.obj'% (
                opt.results_path, opt.name, epoch, test_data['name'])
            gen_mesh(opt, model, cuda, test_data, save_path)


if __name__ == '__main__':
    test(opt)
