from share import *
import os, argparse
import pytorch_lightning as pl
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
import numpy as np
import torch
import torchvision
from torch import autocast
from contextlib import contextmanager, nullcontext
from PIL import Image
from tqdm import tqdm
from dataset.dataset import NSDImageDataset_demo
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# Configs
sd21_path = './ckpt/SD21/control_sd21_ini.ckpt'
checkpoint_path = './ckpt/NSD/finetune_single_sub/sub01/epoch_015.pth'

def recosntruct_images(model, inputs, ddim_sampler, N, ddim_steps, scale, seed, eta):
    with torch.no_grad():
        fmri_code = model.fmri_encoder.encode_feats(inputs)
        control = fmri_code
        control = control.to(memory_format=torch.contiguous_format)

        c, _ = model.cond_stage(fmri_code)

        cond = {"c_concat": [control], "c_crossattn": [c]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_unconditional_conditioning(N, N)]}

        shape = (4, 64, 64)

        samples, intermediates = ddim_sampler.sample(ddim_steps, N,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)

        x_samples = (x_samples.float() + 1) / 2.0
        x_samples = torch.clamp(x_samples, min=0.0, max=1.0).cpu()
    return x_samples


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare output dir
    out_path = './example/demo_results'
    os.makedirs(os.path.join(out_path, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'batch_compare'), exist_ok=True)
    for i in range(opt.n_samples):
        os.makedirs(os.path.join(out_path, 'pred_{}'.format(i)), exist_ok=True)

    # dataset
    test_dataset = NSDImageDataset_demo(data_path='./example/demo_data/example_fmri.npy',
        image_path='./example/demo_data/stimuli_images', train_sub=[1], val_sub=[1], image_norm=True)
    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=opt.batch_size, shuffle=False)

    # create and load model
    model = create_model('./models/inference.yaml').cpu()
    model.model_name = out_path
    model.sd_locked = True
    model.only_mid_control = False

    # load fmri-to-image checkpoint
    if checkpoint_path is not None:
        model_meta = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(model_meta, strict=False)
        print('Resuming from checkpoint: {}'.format(checkpoint_path))

    model = model.to(device)

    ddim_sampler = DDIMSampler(model)
    ddim_steps = opt.ddim_steps
    scale = opt.scale
    seed = opt.seed
    eta = 0.0
    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    # genetation
    model.eval()
    sample_imgs = []
    all_sample_imgs = []
    with torch.no_grad():
        with precision_scope("cuda"):
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Processing batches")):
                gt_image = (batch['gt_image'].float() + 1) / 2.0
                gt_image = torch.clamp(gt_image, min=0.0, max=1.0)
                gt_image = rearrange(gt_image, 'b h w c -> b 1 c h w')
                inputs = batch['fmri'].unsqueeze(1).repeat(1,opt.n_samples,1,1,1)
                inputs = inputs.flatten(0,1).to(device)
                N = inputs.shape[0]
                pred = recosntruct_images(model, inputs, ddim_sampler, N, ddim_steps, scale, seed, eta)
                pred = rearrange(pred, '(b n) c h w -> b n c h w', b=N//opt.n_samples, n=opt.n_samples)

                # save images
                sample_imgs = torch.cat([gt_image, pred], dim=1).flatten(0,1)
                all_sample_imgs.append(sample_imgs)
                for i in range(N // opt.n_samples):
                    # save gt images
                    gt_save = 255. * rearrange(gt_image[i], '1 c h w -> h w c').numpy() 
                    save_path = os.path.join(out_path, 'gt', "bs{:06}-idx-{:06}.png".format(batch_idx, i))
                    Image.fromarray(gt_save.astype(np.uint8)).save(save_path)
                    # save reconstruction
                    for j in range(opt.n_samples):
                        pred_save = 255. * rearrange(pred[i, j], 'c h w -> h w c').numpy() 
                        save_path = os.path.join(out_path, 'pred_{}'.format(j), "bs{:06}-idx-{:06}.png".format(batch_idx, i))
                        Image.fromarray(pred_save.astype(np.uint8)).save(save_path)
                
                grid = torchvision.utils.make_grid(sample_imgs, nrow=opt.n_samples+1)
                grid = 255. * rearrange(grid, 'c h w -> h w c').numpy() 
                save_path = os.path.join(out_path, 'batch_compare', "bs{:06}.jpg".format(batch_idx))
                Image.fromarray(grid.astype(np.uint8)).save(save_path)

            all_sample_imgs = torch.cat(all_sample_imgs, dim=0)
            grid = torchvision.utils.make_grid(all_sample_imgs, nrow=opt.n_rows if opt.n_rows else (opt.n_samples+1)*4)
            grid = 255. * rearrange(grid, 'c h w -> h w c').numpy() 
            save_path = os.path.join(out_path, "result.jpg")
            Image.fromarray(grid.astype(np.uint8)).save(save_path)

    print('The results are saved in {}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given fmri.",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid. default: (n_samples + 1) * 4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    seed = opt.seed
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    main(opt)