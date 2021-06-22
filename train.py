# -*- coding: utf-8 -*-
# author: pinakinathc

import os
import time
import torch
from dataloader import GarmentDataset
from utils import projection, adjust_learning_rate
from model import MultiViewSk
from options import BaseOptions
from torch.utils.data import DataLoader
from torch.utils.tensorboard  import SummaryWriter
from utils import visualise_NDF

# get options
opt = BaseOptions().parse()

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d'%opt.gpu_id)

    train_dataset = GarmentDataset(opt, phase='train')
    test_dataset = GarmentDataset(opt, phase='test')

    # create data loader
    train_data_loader = DataLoader(train_dataset,
        batch_size=opt.batch_size, shuffle=not opt.serial_batches,
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
    summary_writer = SummaryWriter(opt.logdir)

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

    # training
    total_iters = 0
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch, 0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        model.train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            label_tensor = train_data['labels'].to(device=cuda)
            
            # reshape multiview tensor
            image_tensor = image_tensor.view(
                image_tensor.shape[0] * image_tensor.shape[1],
                image_tensor.shape[2], image_tensor.shape[3], image_tensor.shape[4])

            pred, latent_vec = model(image_tensor, sample_tensor)

            # print ('Pred min: {} | Pred max: {} | Pred median: {} | Label min: {} | Label max: {} | Label median: {}'.format(
            #         torch.min(pred), torch.max(pred), torch.median(pred), torch.min(label_tensor), torch.max(label_tensor), torch.median(label_tensor)))

            # error = criterion(torch.clamp(pred, min=0.0, max=0.7), torch.clamp(label_tensor, min=0.0, max=0.7))
            error = criterion(pred, label_tensor)

            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:

                # print ('Error: {} | Pred min: {} | Pred max: {} | Pred median: {} | Label min: {} | Label max: {} | Label median: {}'.format(
                #     error.item(), torch.min(pred), torch.max(pred), torch.median(pred), torch.min(label_tensor), torch.max(label_tensor), torch.median(label_tensor)))

                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader), error.item(), lr, opt.sigma,
                        iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60), int(eta - 60 * (eta // 60))))
                summary_writer.add_scalar('Loss/train', error, total_iters)

            if total_iters % opt.freq_save == 0:
                torch.save(model.state_dict(), '%s/%s/model_latest'% (opt.checkpoints_path, opt.name))

            iter_data_time = time.time()
            total_iters += 1

        # update learning rate
        # lr = adjust_learning_rate(optimizer, epoch, lr, opt.schedule, opt.gamma)

if __name__ == '__main__':
    train(opt)          
