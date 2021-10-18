from __future__ import print_function
import os
import json
import argparse
import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
import torch
import numpy as np
import random
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from collections import OrderedDict
from torch.utils import model_zoo
from tensorboardX import SummaryWriter

from metrics import *
from config import BaseConfig
from data import Dataset
from utils.focal_loss import FocalLoss


def load_pretrained_weights(net, state_dict):
    """ Load the pretrained weights.
        If layers are missing or of  wrong shape, will not load them.
    """
    print("Load model ...")
    new_dict = OrderedDict()
    d = net.state_dict()

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    for k, v in list(state_dict.items()):
        # k = k.replace("backbone.", "") # remove
        if k.startswith('module.encoder_q.'): k = k.replace('module.encoder_q.', '')
        
        if k.startswith('module.'): k = k.replace('module.', '')

        if k not in d:
            print("Skipping ", k)
            continue
        new_dict[k] = v

    # Add missing weights from the network itself
    for k, v in list(d.items()):
        if k not in new_dict:
            if not k.endswith('num_batches_tracked'):
                print("Loading weights for %s: Missing layer %s" % (type(net).__name__, k))
            new_dict[k] = v
        elif v.shape != new_dict[k].shape:
            print("Loading weights for %s: Bad shape for layer %s, skipping" % (type(net).__name__, k))
            new_dict[k] = v


    # print(new_dict)
    net.load_state_dict(new_dict)


def count_total_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("Total Parameters :" + str(k))


def save_model(model, save_path, name, iter_cnt=False):
    if iter_cnt == False:
        save_name = os.path.join(save_path, name + '.pth')
    else:
        save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def validate(val_loader, model, metric_fc, criterion, device, opt):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    total_iter_val = int(val_loader.__len__() / opt.batch_size)
    with torch.no_grad():
        for ii, data in enumerate(val_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()

            acc = np.mean((output == label).astype(int))
            losses.update(loss.item(), data_input.size(0))
            top1.update(acc, data_input.size(0))


        print('Validation : Acc {top1.avg:.3f}, Loss {losses.avg:.3f}'.format(top1=top1, losses=losses))

    return top1.avg, losses.avg


if __name__ == '__main__':
    

    ########### configuration ###########
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-c", "--config", help="configuration file", type=str)
    args, unknown = parser.parse_known_args()
    opt = BaseConfig()

    if args.config:
        config_overide = json.load(open(args.config))
        for key, item in config_overide.items():
            opt.__dict__[key] = item
            print(key, " : ", item)

    # if not opt.name:
    opt.name = opt.name+"_"+opt.backbone+"-"+opt.metric+"-"+opt.loss+"-"+"x".join([str(x) for x in opt.input_shape])
    opt.name = opt.name+"-f"+str(opt.num_feature)+"-c"+str(opt.num_classes) #+"-"+str(int(time.time()))

    save_path = os.path.join(opt.checkpoints_path, opt.name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "interval"), exist_ok=True)

    
    device = torch.device("cuda")

    json.dump(opt.__dict__, open(os.path.join(save_path, "config.json"), "w"))

    if opt.display:
        writer = SummaryWriter(log_dir=save_path)

    ######## Dataset et dataloader ########
    # train dataloader
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True, pin_memory=True,
                                  num_workers=opt.num_workers)

    # val dataloder
    if opt.val_list:
        val_dataset = Dataset(opt.val_root, opt.val_list, phase='val', input_shape=opt.input_shape)
        valloader = data.DataLoader(val_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True, pin_memory=True,
                                    num_workers=opt.num_workers)

    total_iter = int(train_dataset.__len__() / opt.batch_size)

    if opt.loss == 'focal_loss':
        print("Focal Loss")
        criterion = FocalLoss(gamma=2)
    elif opt.loss == 'smoothing':
        print("Label smoothing")
        criterion = LabelSmoothing(smoothing=0.2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    

    model = getattr(__import__('model'), opt.backbone)(opt)

    # print(model)

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(opt.num_feature, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(opt.num_feature, opt.num_classes, s=30, m=opt.margin, easy_margin=opt.easy_margin)
    elif opt.metric == "arc_face":
        metric_fc = ArcfaceModule(embedding_size=opt.num_feature, classnum=opt.num_classes)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(opt.num_feature, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(opt.num_feature, opt.num_classes)

    if opt.load_model_url:
        print(" >>>>>>>>> Load url ")
        load_pretrained_weights(model, model_zoo.load_url(opt.load_model_url))
    if opt.load_model_path:
        print(" >>>>>>>>> Load model ")
        load_pretrained_weights(model, torch.load(opt.load_model_path))
    if opt.load_margin_path:
        print(" >>>>>>>>> Load margin ")
        load_pretrained_weights(metric_fc, torch.load(opt.load_margin_path))

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    # scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=opt.lr_step, gamma=opt.lr_gamma)
    best_acc1 = 0
    start = time.time()

    count_total_params(model)

    
    for i in range(opt.max_epoch):
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                lr_t = scheduler.get_last_lr()
                time_str = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime(time.time()))
                print('{} epoch {} it {} / {} {} it/s loss {} lr {} acc {}'.format(time_str, i, ii, total_iter,
                                                                                   np.around(speed, 2),
                                                                                   np.around(loss.item(), 3), lr_t,
                                                                                   acc))
                with open(os.path.join(save_path, 'train_log.txt'), 'a') as f:
                    f.write('{} epoch {} iter {} {} it/s loss {} lr {} acc {}\n'.format(time_str, i, ii,
                                                                                        np.around(speed, 2),
                                                                                        loss.item(), lr_t, acc))
                if opt.display:
                    writer.add_scalar('Loss/train', loss.item(), iters)
                    writer.add_scalar('Acc/train', acc, iters)

                start = time.time()

        scheduler.step()
        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, os.path.join(save_path, "interval"), opt.backbone, i)
            save_model(metric_fc, os.path.join(save_path, "interval"), "margin", i)

        if opt.val_list:
            val_acc1, val_loss = validate(valloader, model, metric_fc, criterion, device, opt)
            with open(os.path.join(save_path, 'val_log.txt'), 'a') as f:
                f.write('Validation : loss : {} acc : {} \n'.format(val_loss, val_acc1))
            if opt.display:
                writer.add_scalar('Loss/val', val_loss, i)
                writer.add_scalar('Acc/val', val_acc1, i)
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)
            if is_best:
                save_model(model, save_path, "best_" + opt.backbone)
                save_model(metric_fc, save_path, "best_margin")


