# Arcface-pytorch
Pytorch implement of arcface 


# Installation

Install PyTorch and all necessary dependancies : 
```
pip install -r requirements.txt
```

# Quick Start

## How to run a training

1. Create au json configuration file in .config

```
{
    "name" : [Name of experiment],
    "backbone": [backbone],
    "loss": "focal_loss", 
    "batch_size": 64, 
    "num_classes" : [Num class],
    "gpu": 0, 
    "train_root": [Path to data],  
    "train_list": [Patho to csv train set], 

    "val_root": [Path to data], 
    "val_list": [Patho to csv val set (Optional)], 

    "input_shape": [3, 120, 120], 
    "checkpoints_path": "./models/",
    "pretrained": true,
    "pooling": "gem",
    "lr":0.01,
    "lr_step": [33,66,100],
    "max_epoch" : 150,
    "margin": 0.7,
    "save_interval":150,

    "load_model_path":"",
    "load_margin_path": ""
}
```

2. Run training with config
```
python train.py --config=./config/example.json
```

A folder of the experiment name will be create in the "chekpoint_path"


## How to extract embedding from pretrain model

```
python extract_feature.py --config=./models/experiment_name/config.json --path_save=[path to save feature] --list_images=[path to txt file with path of all images]
```


# Citation

```
@inproceedings{deng2018arcface,
    title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
    author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
    booktitle={CVPR},
    year={2019}
}
```