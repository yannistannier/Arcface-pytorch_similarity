import os
from PIL import ImageFilter, Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
# import cv2
import pandas as pd
from tqdm import tqdm
import random

class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(3,224,224), pad_crop = 0):
        self.phase = phase
        self.input_shape = input_shape
        self.pad_crop = pad_crop

        df = pd.read_csv(data_list_file)
        imgs = list(df["name"])

        self.imgs = [os.path.join(root, img) for img in imgs]
        self.labels = [int(l) for l in list(df["label"])]

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((self.input_shape[1], self.input_shape[2])),
                T.RandomRotation((0,90)),
                T.RandomResizedCrop(self.input_shape[1], scale=(0.85, 1.15)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((self.input_shape[1], self.input_shape[2])),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        data = Image.open(self.imgs[index]).convert("RGB")
        data = self.transforms(data)
        label = np.int32(self.labels[index])
        return data.float(), label


    def __len__(self):
        return len(self.imgs)
