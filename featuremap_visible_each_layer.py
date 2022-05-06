"""
作者：didi
日期：2022年05月06日
"""

import os
import wandb
import shutil
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
import torchvision.utils as vutil
from torch.utils.data import DataLoader
import torchsummary

from utils.utils import get_classes, get_anchors
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo4 import YoloBody

COUNT = 0
IMAGE_FOLDER = './save_image'
INSTANCE_FOLDER = None


def hook_func(module, input, output):
    image_name, image_path = get_image_name_for_hook(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)

    from PIL import Image
    from torchvision.utils import make_grid
    grid = make_grid(data, nrow=8, padding=2, pad_value=0.5, normalize=False, range=None, scale_each=False)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    wandb.log({f"{image_name}": wandb.Image(im)})
    im.save(image_path)


def get_image_name_for_hook(module):
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    image_name = '.'  # '.' is surely exist, to make first loop condition True

    global COUNT
    while os.path.exists(image_name):
        COUNT += 1
        image_name = f'{COUNT}_{base_name}'
        image_path = os.path.join(INSTANCE_FOLDER, f'{COUNT}_{base_name}.png')
    return image_name, image_path


if __name__ == '__main__':
    time_beg = time.time()

    # clear output folder
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)

    wandb.init(project="yolov4-pytorch-mchar",
               entity="neverbackdown",
               config={
                   "input_shape": (384, 384),
                   "batch_size": 1,
                   "depth": 50,
                   "Cuda": True
               })
    wandb.watch_called = False
    config = wandb.config  # Initialize config

    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'
    num_classes = len(get_classes(classes_path))
    anchors = get_anchors(anchors_path)

    model = YoloBody(len(anchors[0]), num_classes)
    model_path = "logs/weights/FPN/Epoch_124.pth"

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    print('Finished!')

    val_annotation_path = 'VOC2007/ImageSets/2007_val.txt'
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_val = len(val_lines)

    val_dataset = YoloDataset(val_lines, (config.input_shape[0], config.input_shape[1]), mosaic=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=1, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)

    model.eval()
    modules_for_plot = (torch.nn.LeakyReLU, torch.nn.BatchNorm2d, torch.nn.Conv2d)
    for name, module in model.named_modules():
        if isinstance(module, modules_for_plot):
            module.register_forward_hook(hook_func)

    index = 1
    for idx, batch in enumerate(val_loader):
        # global COUNT
        COUNT = 1

        INSTANCE_FOLDER = os.path.join(IMAGE_FOLDER, f'{index}_pic')

        images, targets = batch[0], batch[1]
        images_val = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
        targets_val = [Variable(torch.from_numpy(ann.astype(float)).type(torch.FloatTensor)) for ann in targets]
        outputs = model(images_val)

        "3 output feature map visible"
        for i, map in enumerate(outputs):
            map = map.clone().detach().permute(1, 0, 2, 3)
            image_name = f'FeatureMap_{i}'
            image_path = os.path.join(INSTANCE_FOLDER, f'{image_name}.png')

            from PIL import Image
            from torchvision.utils import make_grid

            grid = make_grid(map, nrow=8, padding=2, pad_value=0.5, normalize=False, range=None, scale_each=False)
            ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            wandb.log({f"{image_name}": wandb.Image(im)})
            im.save(image_path)

        index += 1
        if index > 10:  # save 20 pictures
            break

