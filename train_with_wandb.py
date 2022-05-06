"""
作者：didi
日期：2022年05月06日
"""


import os
import numpy as np
import wandb
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_classes, get_anchors
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss, Generator
from nets.yolo4 import YoloBody

import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


def fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, cuda):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_size:
            break
        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann.astype(float)).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann.astype(float)).type(torch.FloatTensor)) for ann in targets]
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        total_loss += loss
        waste_time = time.time() - start_time
        if iteration % 10 == 0:
            print(f'iter:{iteration}/{epoch_size} || Total Loss: {total_loss / (iteration + 1)} || {waste_time}/step')

        start_time = time.time()

    print('\nStart Validation')
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_size_val:
            break
        images_val, targets_val = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann.astype(float)).type(torch.FloatTensor)) for ann in
                               targets_val]
            else:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                targets_val = [Variable(torch.from_numpy(ann.astype(float)).type(torch.FloatTensor)) for ann in
                               targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_loss += loss

            if iteration % 10 == 0:
                print(f'iter:{iteration}/{epoch_size_val} || Total Loss: {val_loss / (iteration + 1)}')

    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('--' * 30)
    print('Finish Validation')

    torch.save(model.state_dict(), f'logs/weights/FPN/Epoch_{epoch + 1}.pth')

    # wandb saving
    wandb.log({"training loss": float(total_loss / (epoch_size + 1)),
               "validation loss": float(val_loss / (epoch_size_val + 1))})
    wandb.save(f'model_{epoch}.h5')


if __name__ == "__main__":
    wandb.init(project="yolov4-pytorch-mchar",
               entity="neverbackdown",
               config={
                   "input_shape": (384, 384),
                   "Cosine_lr": False,
                   "freeze_epochs": 50,
                   "epochs": 200,
                   "depth": 50,
                   "freeze_lr": 1e-3,
                   "unfreeze_lr": 1e-4,
                   "Cuda": True,
                   "smoooth_label": 0,
               })
    wandb.watch_called = False
    config = wandb.config  # Initialize config

    annotation_path = 'data/train.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    # create model
    model = YoloBody(len(anchors[0]), num_classes)
    # model_path = "model_data/yolo4_weights.pth"
    # model_path = "model_data/resnest101.pth"
    model_path = "logs/weights/FPN/Epoch_124.pth"

    # pre-trained weight
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # pretrained_dict = {'backbone.' + k: v for k, v in pretrained_dict.items() if np.shape(model_dict['backbone.'+k])==np.shape(v)}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    wandb.watch(model, log="all")
    print('Finished!')

    net = model.train()

    if config.Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes,
                                    (config.input_shape[1], config.input_shape[0]), config.smoooth_label, config.Cuda))

    train_annotation_path = 'VOC2007/ImageSets/2007_train.txt'
    val_annotation_path = 'VOC2007/ImageSets/2007_val.txt'

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        optimizer = optim.Adam(net.parameters(), config.freeze_lr, weight_decay=5e-4)
        if config.Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        train_dataset = YoloDataset(train_lines, (config.input_shape[0], config.input_shape[1]), mosaic=False)
        val_dataset = YoloDataset(val_lines, (config.input_shape[0], config.input_shape[1]), mosaic=False)
        gen = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = max(1, num_train // config.Batch_size)
        epoch_size_val = num_val // config.Batch_size

        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False

        for epoch in range(0, config.freeze_epochs):
            print(f'Epoch:{epoch + 1}')
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, config.Cuda)
            lr_scheduler.step()

    if True:

        optimizer = optim.Adam(net.parameters(), config.unfreeze_lr, weight_decay=5e-4)
        if config.Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        train_dataset = YoloDataset(train_lines, (config.input_shape[0], config.input_shape[1]), mosaic=False)
        val_dataset = YoloDataset(val_lines, (config.input_shape[0], config.input_shape[1]), mosaic=False)
        gen = DataLoader(train_dataset, batch_size=config.batch_size/2, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=config.batch_size/2, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = max(1, num_train // (config.Batch_size/2))
        epoch_size_val = num_val // (config.Batch_size/2)

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(config.freeze_epochs, config.epochs):
            print(f'Epoch:{epoch + 1}')
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, config.Cuda)
            lr_scheduler.step()
