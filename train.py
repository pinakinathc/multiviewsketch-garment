import os
import glob
import torch
from torch.utils.data import DataLoader

from options import opts

if opts.model_name == 'model_A':
    from src.model_A import GarmentModel
elif opts.model_name == 'model_AA':
    from src.model_AA import GarmentModel
elif opts.model_name == 'model_B':
    from src.model_B import GarmentModel
elif opts.model_name == 'model_BB':
    from src.model_BB import GarmentModel
elif opts.model_name == 'model_C':
    from src.model_C import GarmentModel
elif opts.model_name == 'model_D':
    from src.model_D import GarmentModel
elif opts.model_name == 'model_E':
    from src.model_E import GarmentModel
elif opts.model_name == 'model_F':
    from src.model_F import GarmentModel
elif opts.model_name == 'model_G':
    from src.model_G import GarmentModel
elif opts.model_name == 'model_H':
    from src.model_H import GarmentModel
else:
    raise ValueError('opts.model_name option wrong: %s'%opts.model_name)

print ('Using model: %s'%opts.model_name)

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
        use_partial=opts.partial
    )

    dataset_val = GarmentDataset(
        data_dir=opts.data_dir,
        val_path=os.path.join(opts.data_dir, 'val.txt'),
        num_views=opts.num_views,
        num_points=opts.num_points,
        evaluate=True,
        use_partial=opts.partial
    )
    print ('Using partial training: ', opts.partial)

    train_loader = DataLoader(
        dataset=dataset_train, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers
    )

    val_loader = DataLoader(
        dataset=dataset_val, batch_size=opts.batch_size, num_workers=opts.num_workers
    )

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='saved_models',
        filename='%s'%opts.exp_name
    )

    ckpt_path = glob.glob(
        os.path.join('saved_models', '%s*.ckpt'%opts.exp_name))
    if len(ckpt_path) == 0:
        ckpt_path = None
    else:
        ckpt_path = ckpt_path[0]
        print ('resuming training from %s'%ckpt_path)

    trainer = Trainer(
        gpus=-1,
        auto_select_gpus=True,
        benchmark=True,
        val_check_interval=1000,
        max_steps=700000,
        logger=logger,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=ckpt_path
    )

    if ckpt_path is None:
        model = GarmentModel()
    else:
        print ('resuming training from %s'%ckpt_path)
        model = GarmentModel.load_from_checkpoint(ckpt_path)

    # training
    trainer.fit(model, train_loader, val_loader)
    # trainer.validate(model, val_loader)
