# -*- coding: utf-8 -*-

import os
import time
import torch
from options import opts
from src.dataset import GarmentDataset
from src.model_new import GarmentModel as GarmentModel
# from src.model import GarmentModel
from torchvision import transforms

if __name__ == '__main__':

    train_dataset = GarmentDataset(
        data_dir=opts.data_dir,
        val_path=os.path.join(opts.data_dir, 'val.txt'),
        num_views=opts.num_views,
        num_points=opts.num_points,
        # use_partial=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers)

    # os.makedirs(output_dir, exist_ok=True)

    model = GarmentModel()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    # try:
    # 	model.load_model()
    # except:
    # 	print ('Failed to load model from %s -- continuing with random weights'%output_dir)
    
    model.train()
    optimizer = model.configure_optimizers()

    """ Freeze certain modules """
    # model.freeze_module(model.alignNet)
    # model.freeze_module(model.decoder)

    start_time = time.time()
    data_time = 0
    iter_time = 0

    print ('starting training ...')
    for batch_idx, (img_feat, pos_emb_feat, xyz, sdf, all_azi) in enumerate(train_dataloader):
    
        # print ('shape of img: {}, pos_emb: {}, xyz: {}, sdf: {}, all_azi: {}'.format(
        #     img_feat.shape, pos_emb_feat.shape, xyz.shape, sdf.shape, all_azi.shape
        # ))
        num_views = img_feat.shape[1]
        # xyz = xyz.unsqueeze(1).repeat(1, num_views, 1, 1)
        # sdf = sdf.unsqueeze(1).repeat(1, num_views, 1, 1)
        
        """ load data to device """
        img_feat = img_feat.to(device)
        pos_emb_feat = pos_emb_feat.to(device)
        xyz = xyz.to(device)
        sdf = sdf.to(device)
        all_azi = all_azi.to(device)

        data_time += time.time() - start_time

        """ train iteration """
        optimizer.zero_grad()
        loss = model.training_step(
            (img_feat, pos_emb_feat, xyz, sdf, all_azi),
            batch_idx)
        loss.backward()
        optimizer.step()

        iter_time += time.time() - start_time

        if batch_idx % opts.save_freq == 0:
            print ('saving latest model')
            model.save_model()

        if batch_idx % opts.print_freq == 0:
            print ('train:: [%d] loss: %.7f dataT: %.5f iterT: %.5f'%(
                batch_idx, loss, data_time/opts.print_freq, iter_time/opts.print_freq))
            data_time = 0
            iter_time = 0
            start_time = time.time()
