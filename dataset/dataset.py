import json
import cv2
import numpy as np

import os, glob, random, cv2, json, pickle, copy, requests, io, csv, tqdm, h5py
import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
from functools import partial
from PIL import ImageFilter
from collections import Counter
from pathlib import Path
from einops import rearrange


class NSDImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='./NSD/fmri_npy',
        image_path='./NSD/nsddata_stimuli/stimuli/images',
        cap_label_path='./NSD/COCO_73k_annots_curated.npy',
        ann_path='./NSD/COCO_73k_annots_curated.npy',
        train_label='./NSD/unique_triallabel.npy',
        val_label='./NSD/sub1257_shared_triallabel.npy',
        train_sub=[1],val_sub=[1],use_vc=True,
        image_norm=True, random_flip=False, 
        phase='train', val_data_fraction=1.0):
        
        self.image_norm = image_norm
        self.data_path = Path(data_path)
        self.ann = np.load(ann_path,allow_pickle=True) # caption file
        self.data = dict()
        self.labels = dict()
        self.label_dict = dict()
        self.train_sub = train_sub
        self.train_dict_map = None # map index back to subject
        self.val_sub = val_sub
        self.val_dict_map = dict() # map index back to subject
        self.real_images_path = image_path
        self.cap_label = np.load(cap_label_path, allow_pickle=True) # caption file

        train_label = np.load(train_label, allow_pickle=True) # train index for each subject in dict
        self.sub = train_sub if phase == 'train' else val_sub
        self.train_dict_map = dict()
        temp = []
        last=0
        for i in self.sub:
            temp_label = np.array(list(train_label.item()[i-1])) # start from 0, subj index -1
            temp.append(temp_label)
            temp_data = np.array(np.load(self.data_path/f'{i:02d}_label.npy'))
            self.labels[i] = temp_data # transform to numpy array to increase access speed
            label_dict = {}
            for j in range(len(temp_data)):
                label_dict[temp_data[j]] = j
            self.label_dict[i] = label_dict
            for j in range(len(temp_label)):
                self.train_dict_map[j+last]=i
            last +=len(temp_label)
        self.train_label = np.concatenate(temp) 

        val_label_sub1257 = np.load(val_label, allow_pickle=True) # val label for one subject
        # validate using part of the val dataset
        val_label_num = int(val_data_fraction * len(val_label_sub1257))
        val_label_sub1257 = val_label_sub1257[:val_label_num]
        
        temp = []
        last=0
        for i in self.val_sub:
            val_label = []
            for val_lab in val_label_sub1257:
                idx = val_lab
                where_result = np.where(self.labels[i] == idx)[0]
                if len(where_result)>0:  # Check if the condition is met
                    val_label.append(val_lab)
            temp_label = val_label
            temp.append(temp_label)
            for j in range(len(temp_label)):
                self.val_dict_map[j+last]=i
            last += len(temp_label)
        self.val_label = np.concatenate(temp)

        image_transform_list = [transforms.Resize((512, 512))]
        image_transform_list.append(transforms.ToTensor())
        if image_norm:
            image_transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.image_transform = transforms.Compose(image_transform_list)

        if phase == 'train':
            self.is_train=True
        else:
            self.is_train=False
        print(f'Data length:{self.__len__()}')

    def __getitem__(self, i):
        # fmri input
        if self.is_train:
            idx = int(self.train_label[i])
            sub_idx = self.train_dict_map[i]
            surf_idx = self.label_dict[sub_idx][idx]
            surface = np.load(self.data_path/f'{sub_idx:02d}_norm/surf_{surf_idx:06d}.npy')
        else:
            idx = int(self.val_label[i])
            sub_idx = self.val_dict_map[i]
            surf_idx = np.where(self.labels[sub_idx]==idx)[0][0]
            surface = np.load(self.data_path/f'{sub_idx:02d}_norm/surf_{surf_idx:06d}.npy')
        surface = torch.from_numpy(surface)[None]

        # gt images
        image_filename = os.path.join(self.real_images_path, f'image_{idx:06d}.png')
        natural_image = Image.open(image_filename)
        inp_img = self.image_transform(natural_image)
        gt_image = rearrange(inp_img, 'c h w -> h w c')

        # coco caption
        annots = self.cap_label[idx]
        caption = list(annots[annots!=''])
        random_caption = random.choice(caption)

        return {
            "fmri": surface, "txt": random_caption,
            "gt_image": gt_image,
        }

    def __len__(self):
        if self.is_train:
            return  len(self.train_label)
        elif not self.is_train:
            return  len(self.val_label)




class NSDImageDataset_demo(torch.utils.data.Dataset):
    def __init__(self, data_path='./example/demo_data/example_fmri.npy',
        image_path='./example/demo_data/stimuli_images',
        train_sub=[1], val_sub=[1], image_norm=True):
        
        self.image_norm = image_norm
        self.train_sub = train_sub
        self.val_sub = val_sub
        self.real_images_path = image_path

        self.fmri = np.load(data_path)

        image_transform_list = [transforms.Resize((512, 512))]
        image_transform_list.append(transforms.ToTensor())
        if image_norm:
            image_transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.image_transform = transforms.Compose(image_transform_list)

        print(f'Data length:{self.__len__()}')

    def __getitem__(self, i):
        # fmri input
        surface = self.fmri[i]
        surface = torch.from_numpy(surface)

        # gt images
        image_filename = os.path.join(self.real_images_path, f'image_{i:06d}.png')
        natural_image = Image.open(image_filename)
        inp_img = self.image_transform(natural_image)
        gt_image = rearrange(inp_img, 'c h w -> h w c')

        return {
            "fmri": surface, "txt": '',
            "gt_image": gt_image,
        }

    def __len__(self):
        return  len(self.fmri)

