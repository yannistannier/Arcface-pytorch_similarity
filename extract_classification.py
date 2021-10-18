from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import time
from config import BaseConfig
from torch.nn import DataParallel
from tqdm import tqdm
from torch.utils import data
import argparse
from torchvision import transforms as T
# from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

from metrics import *
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"

class ToCenteredLargerBackgroundProjector(object):

    def __call__(self, img):
        # Test if object is numpy or PIL.Image
        img_w, img_h = img.size
        w = 800
        h = 300
        result = Image.new(img.mode, (w,h), 'black')
        result.paste(img,(int((w/2)-(img_w/2)) , int((h/2)-(img_h/2)) ))
        return result


    def __repr__(self):
        return self.__class__.__name__+'()'

class MakeSquareProjector(object):

    def __call__(self, im):
        min_size=224
        fill_color=(0, 0, 0)
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __repr__(self):
        return self.__class__.__name__+'()'


class Dataset(data.Dataset):

    def __init__(self, list_images, input_shape=(3,450, 450), pipeline="imagenet", pad_crop=0):
        self.input_shape = input_shape
        self.imgs = list_images

    
        # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = T.Normalize(mean=[0.408, 0.432, 0.373], std=[0.140, 0.120, 0.108])

        self.transforms = T.Compose([
            # ToCenteredLargerBackgroundProjector(),
            MakeSquareProjector(),
            T.Resize((self.input_shape[1], self.input_shape[2])),
            T.ToTensor(),
            normalize
        ])
 
    def multiscale_transforms(self, data):
        datas = []
        for t in self.transforms_scales:
            datas.append(t.transforms(data).float())
        return datas

    def __getitem__(self, index):
        data = Image.open(self.imgs[index]).convert("RGB")
        data = self.transforms(data)
        return data.float()
        # datas = self.multiscale_transforms(data)
        # return datas

    def __len__(self):
        return len(self.imgs)



def save_feature(model, opt, save_features, list_images):
    with open(list_images) as f:
        list_images = f.read().splitlines()
    list_dataset = Dataset(list_images, opt.input_shape)
    listloader = DataLoader(list_dataset,batch_size=10,shuffle=False, pin_memory=True,num_workers=24)
    features = []
    for data in tqdm(listloader):
        image = data
        # label = np.zeros((image.shape[0], 1)).astype(np.int32)

        output = model(image)
        output = output.data.cpu().numpy()
        # output = np.argmax(output, axis=1)
        
        # output_class = np.argsort(output, axis=1)[:,-5:].astype(np.float)
        # predictions = np.sort(output, axis=1, rev)[::-1][:,-5:].astype(np.float)

        features.extend(output)
    
        # for x in feature:
        #     features.append(x)
        # if features is None:
        #     features = feature
        # else:
        #     features = np.vstack((features, feature))
    features = np.array(features)
    # print(features.shape)
    np.save(save_features, features)


if __name__ == '__main__':
    # time.sleep(4500)
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", '--list_images', type=str, default='')
    parser.add_argument("-c", "--config", help="configuration file", type=str)
    parser.add_argument("-s", "--save_features", help="save feature", type=str)

    args, unknown = parser.parse_known_args()
    opt = BaseConfig()

    if args.config:
        config_overide = json.load(open(args.config))
        for key, item in config_overide.items():
            opt.__dict__[key] = item
            if key in ("name", "backbone", "input_shape"):
                print(key, " : ", item)
    
    name = "classification_"+args.list_images.split("/")[-1][:-4]+"_"+opt.name+"_"+opt.name+".npy"

    if not args.save_features:
        args.save_features = args.list_images.replace(".txt", ".npy")
    
    if args.save_features:
        args.save_features = os.path.join(args.save_features, name)

    
    model_path = args.config.rsplit("/",1)[0]+"/best_"+opt.backbone+".pth"

    # time.sleep(6000)

    model = getattr(__import__('model'), opt.backbone)(opt)


    model = DataParallel(model)


    # # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))




    model.eval()
    save_feature(model, opt, args.save_features, args.list_images)
