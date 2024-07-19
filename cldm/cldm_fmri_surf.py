import einops
import torch
import torch as th
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from functools import partial

from utils.utils import Config
from fmrienc_src.transformer_models import Neural_fMRI2fMRI

import collections
import itertools
import os
import numpy as np
import scipy as sp
import torchvision
from PIL import Image
import lpips
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from dataset.dataset import NSDImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from pytorch_lightning import seed_everything
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


def load_fmri_encoder():
    ckpt_encoder = './ckpt/fMRI2fMRI_UKB/checkpoint_120000.pth'                                                                                                          
    cfg_file = './ckpt/fMRI2fMRI_UKB/fMRI_AutoEncoder.yaml'
    config = Config(cfg_file)

    model_encoder = Neural_fMRI2fMRI(config)

    # load without module
    pretrain_metafile = torch.load(ckpt_encoder, map_location='cpu')
    model_keys = set(model_encoder.state_dict().keys())
    load_keys = set(pretrain_metafile['model'].keys())
    state_dict = pretrain_metafile['model']
    if model_keys != load_keys:
        print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
        if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_encoder.load_state_dict(state_dict)
    print('-----------Loaded FMRI Encoder-----------')

    del model_encoder.transformer.decoder_pos_embed
    del model_encoder.transformer.decoder_blocks
    del model_encoder.transformer.decoder_pred
    del model_encoder.transformer.decoder_embed
    del model_encoder.transformer.decoder_norm

    return model_encoder


class FMRIToSemanticModel(nn.Module):
    """
    High-Level Semantic Feature Learning.
    """

    def __init__(self, cond_dim=1024, fmri_seq_len=257, fmri_latent_dim=1024):
        super(FMRIToSemanticModel, self).__init__()
        self.fmri_seq_len = fmri_seq_len
        self.fmri_latent_dim = fmri_latent_dim
        self.cond_dim = cond_dim

        # Channel Mapper
        self.channel_mapper = nn.Sequential(
            nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
            nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
        )

        # Dimension Mapper
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.norm = nn.LayerNorm(cond_dim)

        # Extra CLIP branch: Parallel processing branch to provide additional semantic information
        self.extra_clip = nn.Sequential(
            nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
            nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True),
            nn.Linear(self.fmri_latent_dim, cond_dim, bias=True),
            nn.LayerNorm(cond_dim)
        )
        self.extra_clip_out = zero_module(nn.Conv1d(77, 77, 3, padding=1))

    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
        x (torch.Tensor): Input tensor representing fMRI data.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output tensor and intermediate tensor (clip_txt).
        """
        latent_crossattn = self.channel_mapper(x)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        clip_txt = self.norm(latent_crossattn)

        extra_clip = self.extra_clip(x)
        extra_clip = self.extra_clip_out(extra_clip)
        out = clip_txt + extra_clip
        
        return out, clip_txt


class FMRIToControlModel(nn.Module):
    """
    Feature Transformation that connects fMRI representation learning and low-level manipulation.
    """

    def __init__(self, fmri_seq_len=257, fmri_latent_dim=1024, out_channel=320, out_dim=64):
        super(FMRIToControlModel, self).__init__()
        self.out_channel = out_channel
        self.out_dim = out_dim
        self.fmri_seq_len = fmri_seq_len
        self.fmri_latent_dim = fmri_latent_dim

        # Channel Mapper
        self.channel_mapper = nn.Sequential(
            nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len * 2, 1, bias=True),
            nn.Conv1d(self.fmri_seq_len * 2, out_channel, 1, bias=True)
        )

        # Dimension Mapper
        self.dim_mapper = nn.Sequential(
            nn.Linear(self.fmri_latent_dim, out_dim ** 2, bias=True),
            nn.SiLU()
        )

        # Output layer: A zero-initialized convolution layer
        self.out = zero_module(conv_nd(1, out_channel, out_channel, 3, padding=1))

    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
        x (torch.Tensor): Input tensor representing fMRI data.
        
        Returns:
        torch.Tensor: Output tensor reshaped to desired dimensions.
        """
        x = self.channel_mapper(x)
        x = self.dim_mapper(x)
        out = self.out(x)
        return out.reshape(-1, self.out_channel, self.out_dim, self.out_dim)


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, mask=False, uniform_tensor=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.fmri_mapper = FMRIToControlModel()
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            # conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            conv_nd(dims, 16, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        print('fmri low-level mapper params', sum(p.numel() for p in self.fmri_mapper.parameters()))

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch


    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.fmri_mapper(hint)

        outs = []

        h = x.type(self.dtype)
        
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class CLDM_for_FMRI(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, 
                 val_perceptual_metrics, sem_loss_weight, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.model_name = None
        self.weight_decay = 0.01
        self.save_interval = 5

        self.mask = False
        self.mask_rate = 0.0

        self.sem_loss_weight = 0.0
        self.uniform_tensor = None

        self.fmri_encoder = load_fmri_encoder()
        self.cond_stage = FMRIToSemanticModel()

        self.val_perceptual_metrics = val_perceptual_metrics
        # Load LPIPS, EfficientNet_B1, Inception_V3 for validation
        if self.val_perceptual_metrics:
            # LPIPS
            self.loss_fn_alex = lpips.LPIPS(net='alex').cpu()
            
            # EfficientNet_B1
            eff_weights = EfficientNet_B1_Weights.DEFAULT
            self.eff_model = create_feature_extractor(efficientnet_b1(weights=eff_weights), 
                                                return_nodes=['avgpool']).cpu()
            self.eff_model.eval().requires_grad_(False)
            self.eff_preprocess = transforms.Compose([
                transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

            inception_weights = Inception_V3_Weights.DEFAULT
            self.inception_model = create_feature_extractor(inception_v3(weights=inception_weights), 
                                                    return_nodes=['avgpool']).cpu()
            self.inception_model.eval().requires_grad_(False)
            self.inception_preprocess = transforms.Compose([
                transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    def get_input(self, batch, k, bs=None, return_unmask=False, *args, **kwargs):
        # encode fmri 2d surface as latent code
        fmri_surf = batch[self.control_key]
        if bs is not None:
            fmri_surf = fmri_surf[:bs]
        fmri_surf = fmri_surf.to(self.device)
        fmri_code = self.fmri_encoder.encode_feats(fmri_surf)
        fmri_code = fmri_code.to(memory_format=torch.contiguous_format).float()
        control = fmri_code.clone()

        # load and encode gt image
        x = super().get_input_gt(batch, self.first_stage_key, *args, **kwargs) 
        
        # High-Level Semantic Feature Learning
        high_latent, pred_clip_feat = self.cond_stage(fmri_code)
        text = batch[self.cond_stage_key]
        
        if self.mask and not return_unmask:
            self.uniform_tensor = torch.rand(high_latent.size(0))
            mask_idx = self.uniform_tensor < self.mask_rate
            mask_batch_size = sum(mask_idx)
            if mask_batch_size > 0:
                uc = self.get_unconditional_conditioning(mask_batch_size)
                high_latent[mask_idx] = uc
        
        if self.sem_loss_weight > 0:
            with torch.no_grad():
                gt_clip_text = self.cond_stage_model.encode_text_final(text)
            # Calculate the MSE loss only for the unmasked elements
            sem_loss = self.sem_loss_weight * torch.nn.functional.mse_loss(gt_clip_text, pred_clip_feat)
            return x, dict(c_crossattn=[high_latent], c_concat=[control], sem_loss=sem_loss)
        else:
            return x, dict(c_crossattn=[high_latent], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N, batch=None):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, x_T=None, exchange=False,
                   **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N, return_unmask=True)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]

        # exchage high level condition 
        if exchange:
            Num = c.shape[0] // 2
            c = torch.cat([c[-Num:], c[:-Num]], dim=0)

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["gtimage"] = batch[self.first_stage_key].permute(0,3,1,2)
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta,
                                                     x_T=x_T)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale >= 1.0:
            uc_cross = self.get_unconditional_conditioning(N, batch)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             x_T=x_T
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log


    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, x_T, **kwargs):
        ddim_sampler = DDIMSampler(self)
        # b, c, h, w = cond["c_concat"][0].shape
        h, w = 512, 512
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, x_T = x_T, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.fmri_encoder.parameters())
        params += list(self.cond_stage.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
            self.fmri_encoder = self.fmri_encoder.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
            self.fmri_encoder = self.fmri_encoder.cuda()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        N = batch[self.control_key].shape[0]
        
        nrows=2
        unconditional_guidance_scale = self.val_scale 
        if unconditional_guidance_scale >= 1.0:
            log = self.log_images(batch, N=N, sample=False, ddim_steps=self.val_ddim_steps, unconditional_guidance_scale=self.val_scale)
            pred = (log[f'samples_cfg_scale_{unconditional_guidance_scale:.2f}'].float() + 1) / 2.0
            pred = torch.clamp(pred, min=0.0, max=1.0).detach()
        else:
            log = self.log_images(batch, N=N, sample=True, ddim_steps=self.val_ddim_steps, unconditional_guidance_scale=self.val_scale)
            pred = (log['samples'].float() + 1) / 2.0
            pred = torch.clamp(pred, min=0.0, max=1.0).detach()
        
        origin = (log['gtimage'].float() + 1) / 2.0
        origin = torch.clamp(origin, min=0.0, max=1.0).detach()
        _, _, _, w = origin.shape
        # we eval results with PSNR, SSIM, LPIPS, effnet_dis, incep_corr
        psnr, ssim, lpips, pixcorr, effnet_dis, incep_corr = [], [], [], [], [], []
        for i in range(N):
            pred_ = pred[i].cpu().numpy()
            origin_ = origin[i].cpu().numpy()
            pixcorr_ = np.corrcoef(origin_.reshape(1, -1), pred_.reshape(1, -1))[0][1]
            mse_ = np.mean((pred_ - origin_ ) ** 2)
            mae_ = np.mean(abs(pred_  - origin_ ))
            psnr_ = 1.0 - 10 * np.log(mse_ + 1e-7) / np.log(10)
            
            pred_np = rgb2gray(pred[i].permute((1,2,0)).cpu())
            origin_np = rgb2gray(origin[i].permute((1,2,0)).cpu())
            ssim_ = structural_similarity(pred_np, origin_np, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            
            pixcorr.append(pixcorr_)
            psnr.append(psnr_)
            ssim.append(ssim_)

            if self.val_perceptual_metrics:
                lpips_ = self.loss_fn_alex(pred[i:i + 1] * 2 - 1., origin[i:i + 1] * 2 - 1.).item()  # lpips needs -1~1
                lpips.append(lpips_)

                gt = self.eff_model(self.eff_preprocess(origin[i:i + 1]))['avgpool']
                gt = gt.to(torch.float32).reshape(1,-1).cpu().numpy()
                fake = self.eff_model(self.eff_preprocess(pred[i:i + 1]))['avgpool']
                fake = fake.to(torch.float32).reshape(1,-1).cpu().numpy()
                effnet_dis_ = sp.spatial.distance.correlation(gt[0], fake[0])
                effnet_dis.append(effnet_dis_)

                gt = self.inception_model(self.inception_preprocess(origin[i:i + 1]))['avgpool']
                fake = self.inception_model(self.inception_preprocess(pred[i:i + 1]))['avgpool']
                fake = fake.float().flatten(1).cpu().numpy()
                gt = gt.float().flatten(1).cpu().numpy()
                incep_corr_ = np.corrcoef(gt, fake)[0,1]
                incep_corr.append(incep_corr_)

        self.log('val/psnr', np.mean(psnr, dtype=np.float64), sync_dist=True, on_epoch=True)
        self.log('val/ssim', np.mean(ssim), sync_dist=True, on_epoch=True)
        self.log('val/pixcorr', np.mean(pixcorr), sync_dist=True, on_epoch=True)

        if self.val_perceptual_metrics:
            self.log('val/lpips', np.mean(lpips), sync_dist=True, on_epoch=True)
            self.log('val/effn_dis', np.mean(effnet_dis), sync_dist=True, on_epoch=True)
            self.log('val/incep_corr', np.mean(incep_corr), sync_dist=True, on_epoch=True)

            result_dict = {
                'psnr': np.mean(psnr), 
                'ssim': np.mean(ssim), 
                'lpips': np.mean(lpips), 
                'pixcorr': np.mean(pixcorr), 
                'effnet_dis': np.mean(effnet_dis), 
                'incep_corr': np.mean(incep_corr),
            }
        else:
            result_dict = {
                'psnr': np.mean(psnr), 
                'ssim': np.mean(ssim), 
                'pixcorr': np.mean(pixcorr), 
            }

        # visualize reconstruction
        sample_imgs = []
        for i in range(N):
            sample_imgs.append(origin[i:i+1])
            sample_imgs.append(pred[i:i+1])
        sample_imgs = torch.cat(sample_imgs, dim=0)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=nrows)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy() 
        filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(self.global_step, self.current_epoch, batch_idx)
        path = os.path.join(self.model_name, 'test', filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid.astype(np.uint8)).save(path)

        return result_dict

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        metric_dict = collections.defaultdict(list)
        for out in outputs:
            for k in out:
                metric_dict[k].append(out[k])

        # Initialize an empty dictionary to store gathered tensors
        gathered_metric_dict = {k: [] for k in metric_dict}

        # Gather tensors from all GPUs
        for k in metric_dict:
            # Convert list of tensors to a single tensor
            tensor_list = np.stack(metric_dict[k])
            gathered_tensors = self.all_gather(tensor_list).cpu().numpy()

            # Flatten the gathered tensors list
            gathered_metric_dict[k] = gathered_tensors.flatten()

        # Only the rank 0 process will print and save the model
        if self.trainer.is_global_zero:
            print('Steps:', self.global_step)
            for k in gathered_metric_dict:
                print(k, np.mean(gathered_metric_dict[k]))
        
        if (self.current_epoch + 1) % self.save_interval == 0 and self.current_epoch != 0 and self.trainer.is_global_zero:
            ckpt_save_dir = os.path.join(self.model_name, 'ckpt')
            os.makedirs(ckpt_save_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_save_dir, f'epoch_{self.current_epoch:03d}_step_{self.global_step:07d}.pth')
            torch.save(self.state_dict(), ckpt_path)
            
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        filename = "b-{:06}.png".format(batch_idx)
        path = os.path.join(self.model_name, 'eval', filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        os.makedirs(os.path.split(path)[0]+'/gt', exist_ok=True)
        os.makedirs(os.path.split(path)[0]+'/pred', exist_ok=True)

        N = batch[self.control_key].shape[0]
        
        nrows=2
        unconditional_guidance_scale = self.val_scale 
        if unconditional_guidance_scale >= 1.0:
            log = self.log_images(batch, N=N, sample=False, ddim_steps=self.val_ddim_steps, unconditional_guidance_scale=self.val_scale)
            pred = (log[f'samples_cfg_scale_{unconditional_guidance_scale:.2f}'].float() + 1) / 2.0
            pred = torch.clamp(pred, min=0.0, max=1.0).detach()
        else:
            log = self.log_images(batch, N=N, sample=True, ddim_steps=self.val_ddim_steps, unconditional_guidance_scale=self.val_scale)
            pred = (log['samples'].float() + 1) / 2.0
            pred = torch.clamp(pred, min=0.0, max=1.0).detach()
        
        origin = (log['gtimage'].float() + 1) / 2.0
        origin = torch.clamp(origin, min=0.0, max=1.0).detach()
        _, _, _, w = origin.shape

        # we eval results with PSNR, SSIM, LPIPS, effnet_dis, incep_corr
        psnr, ssim, lpips, pixcorr, effnet_dis, incep_corr = [], [], [], [], [], []
        for i in range(N):
            pred_ = pred[i].cpu().numpy()
            origin_ = origin[i].cpu().numpy()
            pixcorr_ = np.corrcoef(origin_.reshape(1, -1), pred_.reshape(1, -1))[0][1]
            mse_ = np.mean((pred_ - origin_ ) ** 2)
            mae_ = np.mean(abs(pred_  - origin_ ))
            psnr_ = 1.0 - 10 * np.log(mse_ + 1e-7) / np.log(10)
            
            pred_np = rgb2gray(pred[i].permute((1,2,0)).cpu())
            origin_np = rgb2gray(origin[i].permute((1,2,0)).cpu())
            ssim_ = structural_similarity(pred_np, origin_np, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            
            pixcorr.append(pixcorr_)
            psnr.append(psnr_)
            ssim.append(ssim_)

            if self.val_perceptual_metrics:
                lpips_ = self.loss_fn_alex(pred[i:i + 1] * 2 - 1., origin[i:i + 1] * 2 - 1.).item()  # lpips needs -1~1
                lpips.append(lpips_)

                gt = self.eff_model(self.eff_preprocess(origin[i:i + 1]))['avgpool']
                gt = gt.to(torch.float32).reshape(1,-1).cpu().numpy()
                fake = self.eff_model(self.eff_preprocess(pred[i:i + 1]))['avgpool']
                fake = fake.to(torch.float32).reshape(1,-1).cpu().numpy()
                effnet_dis_ = sp.spatial.distance.correlation(gt[0], fake[0])
                effnet_dis.append(effnet_dis_)

                gt = self.inception_model(self.inception_preprocess(origin[i:i + 1]))['avgpool']
                fake = self.inception_model(self.inception_preprocess(pred[i:i + 1]))['avgpool']
                fake = fake.float().flatten(1).cpu().numpy()
                gt = gt.float().flatten(1).cpu().numpy()
                incep_corr_ = np.corrcoef(gt, fake)[0,1]
                incep_corr.append(incep_corr_)

        self.log('val/psnr', np.mean(psnr, dtype=np.float64), sync_dist=True, on_epoch=True)
        self.log('val/ssim', np.mean(ssim), sync_dist=True, on_epoch=True)
        self.log('val/pixcorr', np.mean(pixcorr), sync_dist=True, on_epoch=True)

        if self.val_perceptual_metrics:
            self.log('val/lpips', np.mean(lpips), sync_dist=True, on_epoch=True)
            self.log('val/effn_dis', np.mean(effnet_dis), sync_dist=True, on_epoch=True)
            self.log('val/incep_corr', np.mean(incep_corr), sync_dist=True, on_epoch=True)

            result_dict = {
                'psnr': np.mean(psnr), 
                'ssim': np.mean(ssim), 
                'lpips': np.mean(lpips), 
                'pixcorr': np.mean(pixcorr), 
                'effnet_dis': np.mean(effnet_dis), 
                'incep_corr': np.mean(incep_corr),
            }
        else:
            result_dict = {
                'psnr': np.mean(psnr), 
                'ssim': np.mean(ssim), 
                'pixcorr': np.mean(pixcorr), 
            }

        # visualize reconstruction
        sample_imgs = []
        for i in range(N):
            sample_imgs.append(origin[i:i+1])
            sample_imgs.append(pred[i:i+1])

            gt_save = 255. * rearrange(origin[i:i+1], '1 c h w -> h w c').cpu().numpy() 
            save_path = os.path.join(os.path.split(path)[0], 'gt', "bs{:06}-idx-{:06}.png".format(batch_idx, i))
            Image.fromarray(gt_save.astype(np.uint8)).save(save_path)

            pred_save = 255. * rearrange(pred[i:i+1], '1 c h w -> h w c').cpu().numpy() 
            save_path = os.path.join(os.path.split(path)[0], 'pred', "bs{:06}-idx-{:06}.png".format(batch_idx, i))
            Image.fromarray(pred_save.astype(np.uint8)).save(save_path)

        sample_imgs = torch.cat(sample_imgs, dim=0)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=nrows)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy() 
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid.astype(np.uint8)).save(path)

        return result_dict

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        metric_dict = collections.defaultdict(list)
        for out in outputs:
            for k in out:
                metric_dict[k].append(out[k])

        if self.local_rank == 0:
            print('Steps:', self.global_step)
            for k in metric_dict:
                print(k, np.mean(metric_dict[k]))
