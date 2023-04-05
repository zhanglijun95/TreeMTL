import os
from PIL import Image
import random

import torch
from torchvision import transforms

class DomainNet(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, h=224, w=224):
        self.dataroot = dataroot
        self.mode = mode
        
        name = 'train.txt' if self.mode == 'train' else 'test.txt'
        with open(os.path.join(dataroot, name), 'r') as f:
            self.img_list = f.readlines()
        self.transform = self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((h, w), scale=(0.6, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        line = self.img_list[item]
        tokens = line.strip().split()
        img_path = tokens[0]
        img_idx = int(tokens[1])
        img = Image.open(os.path.join(self.dataroot, img_path))
        img = img.convert('RGB')
        img_t = self.transform(img)
        return {'img': img_t, 'img_idx': img_idx}

class MultiDomainSampler(torch.utils.data.Sampler):
    def __init__(self, domain_img_list, mode, batch_size, domain_names, random_shuffle):
        super(MultiDomainSampler, self).__init__(domain_img_list)
        self.domain_img_list = domain_img_list[mode]
        self.batch_size = batch_size
        self.domain_names = domain_names
        self.random_shuffle = random_shuffle

        if self.random_shuffle:
            for domain in self.domain_names:
                random.shuffle(self.domain_img_list[domain])

    def __iter__(self):
        for i in range(len(self)):
            batch_idx = []
            for domain in self.domain_names:
                domain_imgs = self.domain_img_list[domain][i * self.batch_size: (i + 1) * self.batch_size]
                batch_idx += domain_imgs
            if i == len(self) - 1 and self.random_shuffle:
                for domain in self.domain_names:
                    random.shuffle(self.domain_img_list[domain])
            yield batch_idx

    def __len__(self):
        l = 1e10
        for domain in self.domain_names:
            d_l = len(self.domain_img_list[domain]) // self.batch_size
            l = min(d_l, l)
        return l