from share import *
import os, argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from dataset.dataset import NSDImageDataset
import numpy as np
import torch

def main(opt):
    # dataset
    val_dataset = NSDImageDataset(train_sub=opt.train_subs, val_sub=opt.val_subs, use_vc=True,
                image_norm=True, random_flip=False,
                phase='val', val_data_fraction=opt.val_data_fraction)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Load models
    model = create_model(opt.config_path).cpu()
    if not opt.model_name:
        train_subs_str = '_'.join(map(str, opt.train_subs))
        mask_flag_str = 'mask' if opt.mask_flag else 'womask'
        opt.model_name = f'sub_{train_subs_str}_{mask_flag_str}'
    model.model_name = os.path.join(opt.outdir, opt.model_name)
    model.load_state_dict(load_state_dict(opt.sd21_path, location='cpu'), strict=False)

    # optimizer setup
    model.learning_rate = opt.learning_rate
    model.weight_decay = opt.weight_decay
    model.sd_locked = True
    model.only_mid_control = False  

    # validation setup
    model.save_interval = opt.save_interval
    model.val_scale = opt.val_scale
    model.val_ddim_steps = opt.val_ddim_steps

    # If true, mask semantic features with unconditional CLIP embeddings corresponding to empty characters
    model.mask = opt.mask_flag
    model.mask_rate = opt.mask_rate

    if opt.checkpoint_path:
        model_meta = torch.load(opt.checkpoint_path, map_location='cpu')
        # model.load_state_dict(model_meta)
        model.load_state_dict(model_meta, strict=False)
        print('Resuming from checkpoint: {}'.format(opt.checkpoint_path))


    precision='bf16'
    trainer = pl.Trainer(gpus=opt.gpu_ids, precision=opt.precision, accumulate_grad_batches=opt.accum_grad,
                         default_root_dir=model.model_name,
                         check_val_every_n_epoch=opt.check_val_every_n_epoch,
                         accelerator='ddp' if len(opt.gpu_ids) > 1 else None,
                         num_nodes=1,
                         max_epochs=opt.max_epochs,
                         )


    # Train!
    trainer.test(model, val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='./models/inference.yaml', 
                        help="path to pretrained checkpoint of model")
    parser.add_argument("--seed", type=int, default=-1, help="the seed")
    cfg = parser.parse_args()

    seed = cfg.seed
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    config_train = OmegaConf.load(cfg.config_path).train
    config_train.config_path = cfg.config_path

    main(config_train)