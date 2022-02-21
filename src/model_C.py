# -*- coding: utf-8 -*-

import os
import torch
from torch.autograd import Variable
from src.networks import AlignUpdater, Decoder, Encoder, AlphaClassifier
import pytorch_lightning as pl

class GarmentModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.freeze_module(self.encoder)
        self.alignUpdater = AlignUpdater()
        self.decoder = Decoder()
        self.alphaClassifier = AlphaClassifier()
        self.latent_feat = torch.nn.Parameter(torch.fmod(
            torch.nn.init.normal_(torch.empty(512), mean=0.0, std=0.02), 2)).cuda()
        self.latent_key = torch.nn.Linear(512, 512)
        self.latent_value = torch.nn.Linear(512, 512)
        self.FF = torch.nn.Linear(512, 512)


        self.softmax = torch.nn.Softmax()
        self.criterion = torch.nn.L1Loss(reduction='none')
        self.criterion_alpha = torch.nn.CrossEntropyLoss()
        self.l1_reg_crit = torch.nn.L1Loss(size_average=False)

    def forward(self, img, pos_emb, xyz):
        """
        Input: 	
                img: Input sketch raster (B x num_views x 3 x 224 x 224)
                pos_emb: Positional Embedding (B x num_views x 10)
                xyz: Points to predict (B x num_views x num_points x 3)
        """
        B, num_views, num_points, _ = xyz.shape
        all_pred_sdf = []
        all_aligned_feat = torch.zeros((B, num_views, 512)).cuda()
        all_alpha = torch.zeros((B, num_views, 512)).cuda()
        all_latent_feat = torch.zeros((B, num_views, 512)).cuda()

        latent_feat = self.latent_feat.repeat(B, 1) # B x 512
        for vid in range(num_views):
            """ Get feature representation from image """
            img_feat = self.encoder(img[:, vid, :, :, :])

            """ Get aligned features from alignNet """
            aligned_feat, alpha = self.alignUpdater(torch.cat([
                img_feat, pos_emb[:, vid, :]], dim=-1))
            all_aligned_feat[:, vid, :] = aligned_feat
            
            """ Combine aligned features using Updater """
            ## Cross Attention
            query = aligned_feat
            key = self.latent_key(latent_feat)
            value = self.latent_value(latent_feat)
            latent_feat = self.FF(torch.matmul(
                self.softmax(torch.matmul(query, key.T) / torch.tensor(512**0.5).cuda()), value))

            ## Re-write in certain location
            attention = torch.nn.functional.hardshrink(alpha+1, lambd=1.0)
            all_alpha[:, vid, :] = attention
            attention = torch.nn.functional.hardtanh(attention, min_val=0.0, max_val=1.0)
            if vid == 0:
                latent_feat = aligned_feat
            else:
                latent_feat = attention*aligned_feat + (1 - attention) * latent_feat
            all_latent_feat[:, vid, :] = latent_feat # Shape of latent_feat: B x 512

            """ Predict SDF using Decoder """
            _, _, num_points, _ = xyz.shape
            combined_feat = torch.cat([
                latent_feat.unsqueeze(1).repeat(1, num_points, 1),
                xyz[:, vid, :, :]], dim=-1).reshape(-1, 512+3)
            pred_sdf = self.decoder(combined_feat)
            all_pred_sdf.append(pred_sdf)

        return all_pred_sdf, all_aligned_feat, all_alpha, all_latent_feat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img, pos_emb, xyz, sdf, mask, all_azi = train_batch
        all_pred_sdf, all_aligned_feat, all_alpha, all_latent_feat = self.forward(img, pos_emb, xyz)
        num_views = sdf.shape[1]

        """ SDF Loss """
        loss = 0
        for vid in range(num_views):
            loss += (vid+1)*(self.criterion(all_pred_sdf[vid].reshape(-1, 1),
                sdf[:, vid, :, :].reshape(-1, 1)) * mask[:, vid, :, :].reshape(-1, 1)).mean()
        self.log('sdf_loss', loss)

        """ Attention Loss """
        loss_alpha = 0
        for vid in range(num_views):
            loss_alpha += self.criterion_alpha(
                self.alphaClassifier(all_alpha[:, vid, :]), all_azi[:, vid])
        self.log('alpha_loss', loss_alpha)
        loss = loss + 0.1*loss_alpha

        """ L1 regularisation Loss """
        loss_reg = 0
        for param in self.alignUpdater.alpha_emb.parameters():
            loss_reg += self.l1_reg_crit(param, target=torch.zeros_like(param))
        self.log('L1_reg', loss_reg)
        loss = loss + 0.0005*loss_reg

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        img, pos_emb, xyz, sdf, mask, all_azi = val_batch
        all_pred_sdf, all_aligned_feat, all_alpha, all_latent_feat = self.forward(img, pos_emb, xyz)
        num_views = sdf.shape[1]

        loss = 0
        for vid in range(num_views):
            loss += (vid+1)*(self.criterion(all_pred_sdf[vid].reshape(-1, 1),
                sdf[:, vid, :, :].reshape(-1, 1)) * mask[:, vid, :, :].reshape(-1, 1)).mean()
        
        self.log('val_sdf_loss', loss)

        loss_alpha = 0
        all_correct = 0
        total = 0
        for vid in range(num_views):
            pred_alpha = self.alphaClassifier(all_alpha[:, vid, :])
            azi = all_azi[:, vid]
            loss_alpha += self.criterion_alpha(pred_alpha, azi)
            correct = (pred_alpha.topk(1, dim=1)[1].reshape(-1) == azi).sum()
            all_correct += correct.item()
            total += azi.shape[0]
        self.log('val_alpha_loss', loss_alpha)
        loss = loss + 0.01*loss_alpha

        self.log('val_loss', loss)
        return all_correct, total

    def validation_epoch_end(self, validation_step_outputs):
        correct = sum([item[0] for item in validation_step_outputs])
        total = sum([item[1] for item in validation_step_outputs])
        self.log('val_alpha_acc', correct/total)

    def freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
