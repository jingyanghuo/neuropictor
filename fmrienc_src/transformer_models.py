import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import numpy as np
from fmrienc_src.transformer_layers import *
from utils.utils import Config
import open_clip


# input: fmri
# fMARI_Reconstruction

# text embedding
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="penultimate"):
        super().__init__()
        assert layer in self.LAYERS

        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokens):
        # tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class MaskGIT_Template(nn.Module):
    def __init__(self, cfg):
        super(MaskGIT_Template, self).__init__()
        self.cfg = cfg
        self.g_config = Config(cfg.VQGAN_CFG)
        cfg.Model['num_codebook'] = self.g_config.Model['num_codebook']

    @torch.no_grad()
    def encode_to_z(self, x):
        return self.vqgan.encode(x)

    @torch.no_grad()
    def z_to_decode(self, x):
        return self.vqgan.decode(x)

    def forward(self, ):
        raise NotImplementedError
    
    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def create_inputs_tokens_normal(self, num, device):
        self.num_latent_size = self.config['resolution'] // self.config['patch_size']
        blank_tokens = torch.ones((num, self.num_latent_size ** 2), device=device)
        masked_tokens = self.mask_token_idx * blank_tokens

        return masked_tokens.to(torch.int64)

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1) # from small to large
        # Obtains cut off threshold given the mask lengths.
        # cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        cut_off = sorted_confidence.gather(dim=-1, index=mask_len.to(torch.long))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    @staticmethod
    def top_k_logits(logits, k):
        v, ix = torch.topk(logits, k, dim=-1)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')

        return out

    @torch.no_grad()
    def predict(self, fmri, image, T=11, mode='cosine'):
        B = image.size(0)
        tgt_indices = self.create_inputs_tokens_normal(B, image.device) # [B, L]

        quant, indices = self.encode_to_z(image) # [B, H, W]
        B, C, H, W = quant.size()

        unknown_number_in_the_beginning = torch.sum(tgt_indices == self.mask_token_idx, dim=-1) # [B]
        gamma = self.gamma_func(mode)
        cur_ids = tgt_indices # [B, L]
        seq_out = []
        mask_out = []

        emb_cls = self.transformer.forward_encoder(fmri) # [B, C]
        for t in range(T):
            logits = self.transformer.forward_decoder(cur_ids, emb_cls) # [B, L, N]
            probs = F.softmax(logits, dim=-1)  # convert logits into probs [B, 256, 1024]
            sampled_ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]
            # _, sampled_ids = torch.topk(probs, k=1, dim=-1) # top-1
            # sampled_ids = sampled_ids.squeeze(-1)

            # v, _ = torch.topk(logits, 20) # multinomial with top-k
            # out = logits.clone()
            # out[out < v[..., [-1]]] = -float('Inf')
            # probs = F.softmax(out, dim=-1)
            # sampled_ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]
            # _, sampled_ids = torch.topk(probs, k=1, dim=-1)
            # sampled_ids = sampled_ids.squeeze(-1)

            unknown_map = (cur_ids == self.mask_token_idx)  # which tokens need to be sampled -> bool [B, 256]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # replace all -1 with their samples and leave the others untouched [B, 256]
            seq_out.append(sampled_ids)
            # seq_out.append(tgt_indices)
            mask_out.append(1. * unknown_map)

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            # selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [B, 257]
            selected_probs = probs.gather(dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)

            selected_probs = torch.where(unknown_map, selected_probs, torch.Tensor([np.inf]).to(logits.device))  # ignore tokens which are already sampled [B, 256]

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....] (B x 1)
            mask_len = torch.maximum(torch.ones_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1, mask_len))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_idx, sampled_ids) # [B, L]

        seq_ids = torch.stack(seq_out, dim=1) # [B, T, L]
        quant = self.vqgan.quantizer.read_codebook(seq_ids.view(-1, 1).repeat(1, C), shape=(B*T, H, W, C)) # [BT, C, H, W]
        quant = quant.view(B, T, C, H, W)
        for i in range(T):
            output = self.z_to_decode(quant[:, i]) # [B, 3, H, W]
            seq_out[i] = output
            mask_out[i] = F.interpolate(mask_out[i].view(B, 1, H, W).repeat(1, 3, 1, 1), size=self.config['resolution'])
        mask = torch.cat(mask_out, dim=-1) # [B, 3, H, TW]
        output = torch.cat(seq_out, dim=-1) # [B, 3, H, TW]

        return mask, output

    # evaluate following the way of training, i.e., random mask a part of GT tokens
    @torch.no_grad()
    def predict2(self, fmri, image, T=11, mode='cosine'):
        quant, indices = self.encode_to_z(image) # [B, H, W]
        B, C, H, W = quant.size()
        indices = indices.flatten(start_dim=1)

        gamma = self.gamma_func(mode)
        seq_out = []
        mask_out = []

        emb_cls = self.transformer.forward_encoder(fmri) # [B, C]
        for t in range(T):
            ratio = 1. * t / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            r = math.floor(mask_ratio * indices.shape[1])
            sample = torch.rand(indices.shape, device=indices.device).topk(r, dim=1).indices
            mask = torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
            mask.scatter_(dim=1, index=sample, value=True)

            masked_indices = self.mask_token_idx * torch.ones_like(indices, device=indices.device)
            z_indices = (~mask) * indices + mask * masked_indices # [B, L, N]

            logits = self.transformer.forward_decoder(z_indices, emb_cls) # [B, L, N]
            probs = F.softmax(logits, dim=-1)  # convert logits into probs [B, 256, 1024]
            sampled_ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]
            # _, sampled_ids = torch.topk(probs, k=1, dim=-1) # top-1
            # sampled_ids = sampled_ids.squeeze(-1)

            sampled_ids = torch.where(mask, sampled_ids, indices)  # replace all -1 with their samples and leave the others untouched [B, 256]
            seq_out.append(sampled_ids)
            mask_out.append(1.*mask)

        seq_ids = torch.stack(seq_out, dim=1) # [B, T, L]
        quant = self.vqgan.quantizer.read_codebook(seq_ids.view(-1, 1).repeat(1, C), shape=(B*T, H, W, C)) # [BT, C, H, W]
        quant = quant.view(B, T, C, H, W)
        for i in range(T):
            output = self.z_to_decode(quant[:, i]) # [B, 3, H, W]
            seq_out[i] = output
            mask_out[i] = F.interpolate(mask_out[i].view(B, 1, H, W).repeat(1, 3, 1, 1), size=self.config['resolution'])
        mask = torch.cat(mask_out, dim=-1) # [B, 3, H, TW]
        output = torch.cat(seq_out, dim=-1) # [B, 3, H, TW]

        return mask, output

    @torch.no_grad()    
    def predict_eval(self, fmri, image, T=11, mode='cosine'):
        B = image.size(0)
        tgt_indices = self.create_inputs_tokens_normal(B, image.device) # [B, L]

        quant, indices = self.encode_to_z(image) # [B, H, W]
        B, C, H, W = quant.size()

        unknown_number_in_the_beginning = torch.sum(tgt_indices == self.mask_token_idx, dim=-1) # [B]
        gamma = self.gamma_func(mode)
        cur_ids = tgt_indices # [B, L]
        seq_out = []

        emb_cls = self.transformer.forward_encoder(fmri) # [B, C]
        for t in range(T):
            logits = self.transformer.forward_decoder(cur_ids, emb_cls) # [B, L, N]
            probs = F.softmax(logits, dim=-1)  # convert logits into probs [B, 256, 1024]
            sampled_ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]
            # _, sampled_ids = torch.topk(probs, k=1, dim=-1) # top-1
            # sampled_ids = sampled_ids.squeeze(-1)

            # v, _ = torch.topk(logits, 20) # multinomial with top-k
            # out = logits.clone()
            # out[out < v[..., [-1]]] = -float('Inf')
            # probs = F.softmax(out, dim=-1)
            # sampled_ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]
            # _, sampled_ids = torch.topk(probs, k=1, dim=-1)
            # sampled_ids = sampled_ids.squeeze(-1)

            unknown_map = (cur_ids == self.mask_token_idx)  # which tokens need to be sampled -> bool [B, 256]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # replace all -1 with their samples and leave the others untouched [B, 256]
            # seq_out.append(sampled_ids)
            # seq_out.append(tgt_indices)

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            # selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [B, 257]
            selected_probs = probs.gather(dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)

            selected_probs = torch.where(unknown_map, selected_probs, torch.Tensor([np.inf]).to(logits.device))  # ignore tokens which are already sampled [B, 256]

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....] (B x 1)
            mask_len = torch.maximum(torch.ones_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1, mask_len))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_idx, sampled_ids) # [B, L]

        quant = self.vqgan.quantizer.read_codebook(sampled_ids.view(-1, 1).repeat(1, C), shape=(B, H, W, C)) # [B, C, H, W]
        output = self.z_to_decode(quant)
        return output

  
class Neural_fMRI2fMRI(nn.Module):
    def __init__(self, cfg):
        super(Neural_fMRI2fMRI, self).__init__()

        self.transformer = fMRI_Autoencoder(cfg)

    def forward(self, image, mask=None):
        # fmri: [B, 3, H, W]
        B = image.size(0)

        # transformer
        rec, loss = self.transformer(image, mask) # [B, L, p*p*3]

        return rec, loss 
   
    def encode_feats(self, image):
        # fmri: [B, 3, H, W]
        B, _, H, W = image.size()

        # transformer
        # feats = self.transformer.forward_encoder(image) # [B, 1, C]
        feats = self.transformer.forward_encoder_wo_pred(image) # [B, 77, C]

        return feats


class Neural_fMRI2fMRI_AR(MaskGIT_Template):
    def __init__(self, cfg):
        super(Neural_fMRI2fMRI_AR, self).__init__(cfg)
        self.config = cfg.Model

        self.transformer = fMRI_Encoder_Decoder(cfg)
        self.vqgan = MaskGIT_VQModel2(self.g_config.Model).eval()

    def forward(self, image):
        # fmri: [B, 3, H, W]
        B, _, H, W = image.size()

        # quantization
        _, indices = self.encode_to_z(image) # [B, H, W]
        indices = indices.flatten(start_dim=1) # [B, HW]

        # transformer
        pred = self.transformer(indices) # [B, L, N]
        target = indices # [B, L]

        return pred, target

    @torch.no_grad()
    def encode_feats(self, image):
        # fmri: [B, 3, H, W]
        B, _, H, W = image.size()

        # quantization
        quant, indices = self.encode_to_z(image) # [B, H, W]
        indices = indices.flatten(start_dim=1) # [B, HW]

        # transformer
        feats = self.transformer.forward_encoder(indices) # [B, 1, C]

        return feats, quant.size()

    @torch.no_grad()
    def decode_fmri(self, latent, quant_size=None, temp=1.0, topk=None, greed=True):

        logits = self.transformer.forward_decoder(latent) # [B, L, N]
        logits = logits / temp # [B, L, N]
        if topk is not None:
            logits = self.top_k_logits(logits, topk)
        probs = F.softmax(logits, dim=-1)  # [B, L, N]
        # sample from the distribution or take the most likely
        if greed:
            _, ids = torch.topk(probs, k=1, dim=-1) # [B, L, 1]
            ids = ids.squeeze(-1)
        else:
            ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]

        B = latent.shape[0]
        if quant_size is not None:
            _, C, H, W = quant_size
        else:
            raise ValueError
        quant = self.vqgan.quantizer.read_codebook(ids.view(-1, 1).repeat(1, C), shape=(B, H, W, C)) # [B, C, H, W]
        output = self.z_to_decode(quant) # [B, 3, H, W]

        return output
    
    @torch.no_grad()
    def predict(self, image, temp=1.0, topk=None, greed=True):
        quant, indices = self.encode_to_z(image) # [B, C, H, W], [B, H, W]
        B, C, H, W = quant.size()
        indices = indices.flatten(start_dim=1)

        logits = self.transformer(indices) # [B, L, N]
        logits = logits / temp # [B, L, N]
        if topk is not None:
            logits = self.top_k_logits(logits, topk)
        probs = F.softmax(logits, dim=-1)  # [B, L, N]
        # sample from the distribution or take the most likely
        if greed:
            _, ids = torch.topk(probs, k=1, dim=-1) # [B, L, 1]
            ids = ids.squeeze(-1)
        else:
            ids = torch.distributions.categorical.Categorical(probs=probs).sample() # [B, L]

        quant = self.vqgan.quantizer.read_codebook(ids.view(-1, 1).repeat(1, C), shape=(B, H, W, C)) # [B, C, H, W]
        output = self.z_to_decode(quant) # [B, 3, H, W]

        return output


class customized_CLIP(nn.Module):
    def __init__(self, cfg):
        super(customized_CLIP, self).__init__()
        
        self.model, _, _ = open_clip.create_model_and_transforms(cfg['clip_name'], pretrained=cfg['clip_ckpt'], cache_dir=cfg['clip_cache'])
        # self.tokenizer = open_clip.get_tokenizer(cfg['clip_name'])
    
    def encode_text(self, text, norm):
        # text = self.tokenizer(text)
        text_features = self.model.encode_text(text, norm) 

        return text_features

    def encode_image(self, image, norm):
        image_features = self.model.encode_image(image, norm)

        return image_features

    def forward(self, ):
        pass




