import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

import open_clip
from ldm.util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        print(arch, version)
        # # cache_dir = ''
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version, cache_dir=cache_dir)
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

    @torch.no_grad()
    def encode_text_final(self, text):
        tokens = open_clip.tokenize(text).to(self.device)
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)

        return x

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
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


class FrozenOpenCLIP_T_V(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        print(arch, version)
        # cache_dir = ''
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version, cache_dir=cache_dir)
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

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    @torch.no_grad()
    def forward_vis(self, x: torch.Tensor):
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.visual.ln_post(x)
        return x

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
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

    @torch.no_grad()
    def cal_loss(self, x, img):
        x = x[torch.arange(x.shape[0]), -1] @ self.model.text_projection
        text_features = F.normalize(text_features, dim=-1)

        img_features = self.model.forward_vis(img)
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=x.device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        return total_loss



class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class PromptOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        print(arch, version)
        # cache_dir = ''
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version, cache_dir=cache_dir)
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
        
        with torch.no_grad():
            # init_text = 'a highly detailed photo with best quality and vivid colors '
            # init_text = 'restore the image'
            init_text = ''
            init_tokens = open_clip.tokenize(init_text)
            init_embed_txt = self.model.token_embedding(init_tokens)
            init_embed_txt = self.encode_with_transformer(init_embed_txt)

        self.embed_txt = nn.Parameter(init_embed_txt)

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        z = self.embed_txt.repeat(len(text), 1, 1)
        return z

    def encode_with_transformer(self, x):
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
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


class PromptEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        print(arch, version)
        # cache_dir = ''
        self.embed_txt = nn.Parameter(torch.randn(1, 77, 1024))

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
        # self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        z = self.embed_txt.repeat(len(text), 1, 1)
        return z

    def encode(self, text):
        return self(text)




from typing import Union, List

import open_clip
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


def init_special_embeddings(tokenizer, special_tokens, model, init_text, tokenwise_init):
    # special initialization
    sp_emb_weights = torch.zeros((len(special_tokens), model.token_embedding.embedding_dim), dtype=torch.float32)
    if tokenwise_init:  # init the embedding with splited sentence tokens
        origin_tokens = tokenizer.encode(init_text[0])[:len(special_tokens)]
        for i, tok_idx in enumerate(origin_tokens):
            sp_emb_weights[i] = model.token_embedding.weight[tok_idx, :].detach()
        for i in range(len(origin_tokens), len(special_tokens)):
            init_feats = []
            mean_tokens = tokenizer.encode(init_text[i])
            for tok_idx in mean_tokens:
                init_feats.append(model.token_embedding.weight[tok_idx:tok_idx + 1, :].detach())
            init_feats = torch.mean(torch.stack(init_feats, dim=0), dim=0)
            sp_emb_weights[i] = init_feats
    else:
        for i, sp_token in enumerate(special_tokens):
            # we first get token index of the original tokens
            init_feats = []
            if init_text is None:
                origin_tokens = tokenizer.encode(sp_token.strip('<').strip('>').replace('-', ' '))
            else:
                origin_tokens = tokenizer.encode(init_text[i])
            for tok_idx in origin_tokens:
                init_feats.append(model.token_embedding.weight[tok_idx:tok_idx + 1, :].detach())
            init_feats = torch.mean(torch.stack(init_feats, dim=0), dim=0)
            sp_emb_weights[i] = init_feats

    return sp_emb_weights


def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<start_of_text>"]
    eot_token = tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class PromptCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77, freeze=True, layer="last",
                 special_tokens=['<left>', '<right>'], init_text='<random>', tokenwise_init=False, deep_prompt=False, cross_attn_layers=16,
                 **kwargs):
        super().__init__()
        assert layer in self.LAYERS
        # model, _, preprocess_val = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version, cache_dir='../moving/ckpt')
        # cache_dir = ''
        model, _, preprocess_val = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version, cache_dir=cache_dir)
        self.visual_preprocess = preprocess_val
        # del model.visual
        self.deep_prompt = deep_prompt
        self.cross_attn_layers = cross_attn_layers

        if special_tokens[0].startswith('repeat_'):  # if start with "repeat-{n}", we need to adjust each sp_token with the index
            n = int(special_tokens[0].split('_')[1])
            special_tokens = list(special_tokens)
            init_text = list(init_text)
            special_tokens = special_tokens * n
            init_text = init_text * n
            for i in range(n):
                special_tokens[i] = special_tokens[i].split('_')[-1].replace('>', f'{i}>')
        special_tokens = list(special_tokens)

        if deep_prompt:  # further repeat prompt for different model layers
            deep_special_tokens = []
            for layer_i in range(cross_attn_layers):
                special_tokens_ = [t.replace('>', f'-layer{layer_i}>') for t in special_tokens]
                deep_special_tokens.extend(special_tokens_)
            special_tokens = deep_special_tokens
            init_text = init_text * cross_attn_layers

        self.special_tokens = special_tokens
        self.tokenizer = open_clip.SimpleTokenizer(special_tokens=special_tokens)
        self.vocab_size = model.vocab_size  # vocab_size=49408, <start> is 49406, <end> is 49407

        if init_text[0] == "<random>":  # random from \mathcal{N}(0, 1)
            print('!!!!!!!!! random init embedding:', len(special_tokens), model.token_embedding.embedding_dim, '!!!!!!!!!!')
            self.special_embeddings = nn.Embedding(len(special_tokens), embedding_dim=model.token_embedding.embedding_dim)
        else:
            # initialization from model's embedding
            sp_emb_weights = init_special_embeddings(self.tokenizer, special_tokens, model, init_text, tokenwise_init)
            print('!!!!!!!!! new embedding shape:', sp_emb_weights.shape, '!!!!!!!!!!')
            self.special_embeddings = nn.Embedding(len(special_tokens), embedding_dim=model.token_embedding.embedding_dim, _weight=sp_emb_weights)
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

    def forward(self, text):
        B, nlayer, L = None, None, None
        if self.deep_prompt:
            tokens = []
            for text_ in text:
                tokens_ = tokenize(self.tokenizer, text_)  # [B,L]
                tokens.append(tokens_)
            tokens = torch.stack(tokens, dim=1)  # [B,nlayer,L]
            B, nlayer, L = tokens.shape
            tokens = tokens.reshape(B * nlayer, L)
        else:
            tokens = tokenize(self.tokenizer, text)  # [B,L]

        # special tokens 赋值>max_emb的数, 对normal_tokens clip, new tokens clip - max_emb, 并且获取mask
        token_mask = (tokens >= self.vocab_size).to(torch.long).unsqueeze(-1).to(self.device)
        tokens_regular = torch.clamp(tokens, 0, self.vocab_size - 1).to(self.device)
        tokens_special = torch.clamp_min(tokens.clone() - self.vocab_size, 0).to(self.device)
        emb_regular = self.model.token_embedding(tokens_regular)
        emb_special = self.special_embeddings(tokens_special)

        text_emb = emb_regular * (1 - token_mask) + emb_special * token_mask
        z = self.encode_with_transformer(text_emb.to(self.device))  # [B(B*nlayer),L,C]

        if self.deep_prompt:
            z = z.reshape(B, nlayer, L, -1)
        return z

    def encode_with_transformer(self, x):
        # x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
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

