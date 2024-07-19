#!/usr/bin/env python
# coding: utf-8

"""
This code is modified from [MindEye]
Original source: [https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/Reconstruction_Metrics.py]
"""

import os
# Set the cache folder path to a custom location
# os.environ['TORCH_HOME'] = '/your/path/to/.cache/torch'
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
import argparse
from PIL import Image
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, 
        default='./infer_results/finetune_single_sub/sub1_epoch_015_ucond_scale5.0',
        help="root dir of the recon results",)
opt = parser.parse_args()

root_dir = opt.root_dir


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

seed=42
seed_everything(seed=seed)

pred_dir = [pre for pre in os.listdir(root_dir) if 'pred_' in pre]

data = {}
data["Metric"] = ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"]

for pred_path in pred_dir:
    images_dir = os.path.join(root_dir, 'gt')
    img_paths = sorted(os.listdir(images_dir))
    all_images = []
    for img_path in img_paths:
        img = Image.open(os.path.join(images_dir, img_path))
        img = transforms.ToTensor()(img)
        all_images.append(img)
    all_images = torch.stack(all_images)

    images_dir = os.path.join(root_dir, pred_path)

    img_paths = sorted(os.listdir(images_dir))
    all_brain_recons = []
    for img_path in img_paths:
        img = Image.open(os.path.join(images_dir, img_path))
        img = transforms.ToTensor()(img)
        all_brain_recons.append(img)
    all_brain_recons = torch.stack(all_brain_recons)

    print('gt image shape: ', all_images.shape)
    print('recon image shape: ', all_brain_recons.shape)

    img_num = all_images.shape[0]

    all_images = all_images.to(device)
    all_brain_recons = all_brain_recons.to(device).to(all_images.dtype).clamp(0,1)


    imsize = 256
    all_images = transforms.Resize((imsize,imsize))(all_images)
    all_brain_recons = transforms.Resize((imsize,imsize))(all_brain_recons)

    np.random.seed(0)

    # # 2-Way Identification

    from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

    def cosine_similarity(vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    @torch.no_grad()
    def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
        preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
        reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
        if feature_layer is None:
            preds = preds.float().flatten(1).cpu().numpy()
            reals = reals.float().flatten(1).cpu().numpy()
        else:
            preds = preds[feature_layer].float().flatten(1).cpu().numpy()
            reals = reals[feature_layer].float().flatten(1).cpu().numpy()
        # if feature_layer is None:
        #     preds = torch.stack([model(preprocess(recon).unsqueeze(0).to(device)) for recon in all_brain_recons], dim=0)
        #     reals = torch.stack([model(preprocess(indiv).unsqueeze(0).to(device)) for indiv in all_images], dim=0)
        # else:
        #     preds = torch.stack([model(preprocess(recon).unsqueeze(0).to(device))[feature_layer] for recon in all_brain_recons], dim=0)
        #     reals = torch.stack([model(preprocess(indiv).unsqueeze(0).to(device))[feature_layer] for indiv in all_images], dim=0)
        # preds = preds.float().flatten(1).cpu().numpy()
        # reals = reals.float().flatten(1).cpu().numpy()
        
        r = np.corrcoef(reals, preds)

        r = r[:len(all_images), len(all_images):]
        congruents = np.diag(r)

        success = r < congruents
        success_cnt = np.sum(success, 0)

        if return_avg:
            perf = np.mean(success_cnt) / (len(all_images)-1)
            return perf
        else:
            return success_cnt, len(all_images)-1

    # ## PixCorr
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()

    corrsum = 0
    for i in tqdm(range(img_num)):
        corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
    corrmean = corrsum / img_num

    pixcorr = corrmean


    # ## SSIM
    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
    print("converted, now calculating ssim...")

    ssim_score=[]
    for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
        ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    ssim = np.mean(ssim_score)


    # ### AlexNet

    from torchvision.models import alexnet, AlexNet_Weights
    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    layer = 'early, AlexNet(2)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.4')
    alexnet2 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet2:.4f}")

    layer = 'mid, AlexNet(5)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.11')
    alexnet5 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet5:.4f}")


    # ### InceptionV3

    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            inception_model, preprocess, 'avgpool')
            
    inception = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {inception:.4f}")


    # ### CLIP

    import clip
    clip_model, preprocess = clip.load("ViT-L-14.pt", device=device)
    clip_model.eval()

    preprocess = transforms.Compose([
        # transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            clip_model.encode_image, preprocess, None) # final layer
    clip_ = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {clip_:.4f}")


    # ### Efficient Net

    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool']).to(device)
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:", effnet)


    # ### SwAV

    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",swav)


    data[pred_path] = [pixcorr, ssim, alexnet2, alexnet5, inception, clip_, effnet, swav]

df = pd.DataFrame(data)
print(df.to_string(index=False))

save_path = os.path.join(root_dir, 'metric.csv')
df.to_csv(save_path, sep='\t', index=False)
print('The metric are saved in {}'.format(save_path))
        

