# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from src.networks import AlignUpdater, Decoder, Encoder, AlphaClassifier
import pytorch_lightning as pl

class GarmentModel(pl.LightningModule):
    def __init__(self, output_dir='output_all/'):
        super().__init__()
        
        self.encoder = Encoder()
        self.freeze_module(self.encoder)
        self.alignUpdater = AlignUpdater()
        self.decoder = Decoder()
        self.alphaClassifier = AlphaClassifier()
        self.output_dir = output_dir
        self.load_model()

        self.criterion = torch.nn.L1Loss()
        self.criterion_alpha = torch.nn.CrossEntropyLoss()

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
        all_alpha = torch.zeros((B, num_views, 512)).cuda()
        all_latent_feat = torch.zeros((B, num_views, 512)).cuda()

        for vid in range(num_views):
            """ Get feature representation from image """
            img_feat = self.encoder(img[:, vid, :, :, :])

            """ Get aligned features from alignNet """
            aligned_feat, alpha = self.alignUpdater(torch.cat([
                img_feat, pos_emb[:, vid, :]], dim=-1))
            all_aligned_feat[:, vid, :] = aligned_feat
            all_alpha[:, vid, :] = alpha
            
            """ Combine aligned features using Updater """
            if vid == 0:
                latent_feat = aligned_feat.clone()
            else:
                # latent_feat = aligned_feat
                latent_feat = alpha * aligned_feat + (1-alpha)*latent_feat
            all_latent_feat[:, vid, :] = latent_feat # Shape of latent_feat: B x 512

            """ Predict SDF using Decoder """
            _, _, num_points, _ = xyz.shape
            combined_feat = torch.cat([
                latent_feat.unsqueeze(1).repeat(1, (vid+1)*num_points, 1),
                torch.cat([xyz[:, i, :, :] for i in range(vid+1)], dim=1)], dim=-1).reshape(-1, 512+3)
            pred_sdf = self.decoder(combined_feat)
            # all_pred_sdf[:, vid, :] = pred_sdf
            all_pred_sdf.append(pred_sdf)

        return all_pred_sdf, all_aligned_feat, all_alpha, all_latent_feat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img, pos_emb, xyz, sdf, all_azi = train_batch
        all_pred_sdf, all_aligned_feat, all_alpha, all_latent_feat = self.forward(img, pos_emb, xyz)
        num_views = sdf.shape[1]

        loss = 0
        for vid in range(num_views):
            loss += self.criterion(all_pred_sdf[vid].reshape(-1, 1),
                torch.cat([sdf[:, i, :, :] for i in range(vid+1)], dim=1).reshape(-1, 1))

        self.log('sdf_loss', loss)

        loss_alpha = 0
        for vid in range(num_views):
            loss_alpha += self.criterion_alpha(
                self.alphaClassifier(all_alpha[:, vid, :]), all_azi[:, vid])
        self.log('alpha_loss', loss_alpha)

        loss = loss + 0.01*loss_alpha
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        img, pos_emb, xyz, sdf, all_azi = val_batch
        all_pred_sdf, all_aligned_feat, all_alpha, all_latent_feat = self.forward(img, pos_emb, xyz)
        num_views = sdf.shape[1]

        loss = 0
        for vid in range(num_views):
            loss += self.criterion(all_pred_sdf[vid].reshape(-1, 1),
                torch.cat([sdf[:, i, :, :] for i in range(vid+1)], dim=1).reshape(-1, 1))
        
        self.log('val_sdf_loss', loss)

        loss_alpha = 0
        for vid in range(num_views):
            loss_alpha += self.criterion_alpha(
                self.alphaClassifier(all_alpha[:, vid, :]), all_azi[:, vid])
        self.log('val_alpha_loss', loss_alpha)

        loss = loss + 0.01*loss_alpha
        self.log('val_loss', loss)
        return loss

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        torch.save(self.alignUpdater.state_dict(), os.path.join(output_dir, 'alignUpdater.net'))
        torch.save(self.alphaClassifier.state_dict(), os.path.join(output_dir, 'alignClassifier.net'))
        torch.save(self.decoder.state_dict(), os.path.join(output_dir, 'decoder.net'))

    def load_model(self, output_dir=None):
        print ('loading saved model')
        if output_dir is None:
            output_dir = self.output_dir
        self.alignUpdater.load_state_dict(torch.load(
            os.path.join(output_dir, 'alignUpdater.net')))
        self.alphaClassifier.load_state_dict(torch.load(
            os.path.join(output_dir, 'alignClassifier.net')
        ))
        self.decoder.load_state_dict(torch.load(
            os.path.join(output_dir, 'decoder.net')))

    def freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
