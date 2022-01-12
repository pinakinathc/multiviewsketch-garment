# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from src.networks import AlignNet, Updater, Decoder, Encoder
import pytorch_lightning as pl

class GarmentModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.freeze_module(self.encoder)
        self.alignNet = AlignNet()
        self.updater = Updater()
        self.decoder = Decoder()

        # # Load prior saved models
        # self.alignNet.load_state_dict(torch.load(
        # 	os.path.join('output', 'align.net')))
        # self.updater.load_state_dict(torch.load(
        # 	os.path.join('output', 'updater.net')))
        # self.decoder.load_state_dict(torch.load(
        # 	os.path.join('output', 'decoder.net')))

        self.criterion = torch.nn.L1Loss()

    def forward(self, img, pos_emb, xyz):
        """
        Input: 	
                img: Input sketch raster (B x num_views x 3 x 224 x 224)
                pos_emb: Positional Embedding (B x num_views x 10)
                xyz: Points to predict (B x num_views x num_points x 3)
        """
        B, num_views, num_points, _ = xyz.shape
        # all_pred_sdf = torch.zeros((B, num_views, num_points, 1)).cuda()
        all_pred_sdf = []
        all_aligned_feat = torch.zeros((B, num_views, 512)).cuda()
        all_latent_feat = torch.zeros((B, num_views, 512)).cuda()

        for vid in range(num_views):
            """ Get feature representation from image """
            img_feat = self.encoder(img[:, vid, :, :, :])

            """ Get aligned features from alignNet """
            aligned_feat = self.alignNet(torch.cat([
                img_feat, pos_emb[:, vid, :]], dim=-1))
            all_aligned_feat[:, vid, :] = aligned_feat
            
            """ Combine aligned features using Updater """
            if vid == 0:
                latent_feat = aligned_feat.clone()
            else:
                latent_feat = self.updater(latent_feat, aligned_feat, pos_emb[:, vid, :])
            all_latent_feat[:, vid, :] = latent_feat # Shape of latent_feat: B x 512

            """ Predict SDF using Decoder """
            _, _, num_points, _ = xyz.shape
            combined_feat = torch.cat([
                latent_feat.unsqueeze(1).repeat(1, num_points, 1),
                xyz[:, vid, :, :]], dim=-1).reshape(-1, 512+3)
            pred_sdf = self.decoder(combined_feat)
            # all_pred_sdf[:, vid, :] = pred_sdf
            all_pred_sdf.append(pred_sdf)

        return all_pred_sdf, all_aligned_feat, all_latent_feat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img, pos_emb, xyz, sdf, mask, all_azi = train_batch
        all_pred_sdf, all_aligned_feat, all_latent_feat = self.forward(img, pos_emb, xyz)
        num_views = sdf.shape[1]

        loss = 0
        for vid in range(num_views):
            loss += self.criterion(all_pred_sdf[vid].reshape(-1, 1), sdf[:, vid, :, :].reshape(-1, 1))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        img, pos_emb, xyz, sdf, mask, all_azi = val_batch
        all_pred_sdf, all_aligned_feat, all_latent_feat = self.forward(img, pos_emb, xyz)
        num_views = sdf.shape[1]

        loss = 0
        for vid in range(num_views):
            loss += self.criterion(all_pred_sdf[vid].reshape(-1, 1), sdf[:, vid, :, :].reshape(-1, 1))

        self.log('val_loss', loss)
        return loss

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = 'output'
        torch.save(self.alignNet.state_dict(), os.path.join(output_dir, 'align.net'))
        torch.save(self.updater.state_dict(), os.path.join(output_dir, 'updater.net'))
        torch.save(self.decoder.state_dict(), os.path.join(output_dir, 'decoder.net'))

    def load_model(self, output_dir=None):
        if output_dir is None:
            output_dir = 'output'
        self.alignNet.load_state_dict(torch.load(os.path.join(output_dir, 'align.net')))
        self.updater.load_state_dict(torch.load(os.path.join(output_dir, 'updater.net')))
        self.decoder.load_state_dict(torch.load(os.path.join(output_dir, 'decoder.net')))

    def freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
