import os
import torch
from torch.utils.data import DataLoader

from options import opts
# from src.model import GarmentModel
from src.model_new import GarmentModel as GarmentModel
from src.dataset import GarmentDataset

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':

    dataset_train = GarmentDataset(
        data_dir=opts.data_dir,
        val_path=os.path.join(opts.data_dir, 'val.txt'),
        num_views=opts.num_views,
        num_points=opts.num_points,
        evaluate=False,
        # use_partial=True
    )

    dataset_val = GarmentDataset(
        data_dir=opts.data_dir,
        val_path=os.path.join(opts.data_dir, 'val.txt'),
        num_views=opts.num_views,
        num_points=opts.num_points,
        evaluate=True,
        # use_partial=True
    )

    train_loader = DataLoader(
        dataset=dataset_train, batch_size=opts.batch_size,
        shuffle=True, num_workers=12
    )

    val_loader = DataLoader(
        dataset=dataset_val, batch_size=opts.batch_size, num_workers=12
    )

    logger = TensorBoardLogger('tb_logs_lab', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='saved_models_lab',
        save_top_k=3,
        filename='%s-{val_loss:.2}'%opts.exp_name
    )

    trainer = Trainer(
        gpus=-1,
        auto_select_gpus=True,
        benchmark=True,
        val_check_interval=1000,
        max_steps=500000,
        # accumulate_grad_batches=8,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    model = GarmentModel()

    # training
    trainer.fit(model, train_loader, val_loader)
    # trainer.validate(model, val_loader)
