"""
Function Storage
"""

### Import
import logging
import datetime
import time
import random
import sys
import shutil
import os
import torch
import monai
import matplotlib.pyplot as plt
import skimage.io as im
from torch.utils.tensorboard import SummaryWriter
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    RandFlip,
    ResizeWithPadOrCrop
)

### Load Data
def load_data_labels(root_path="./data/train"):
    """
    :param root_path: str, root of images (train / test)
    :return: lists of image file names and labels (0: benign, 1: malignant)
    """
    # compile data files & labels
    images = []
    labels = []
    for type in ["benign", "malignant"]:
        curr_data = [os.path.join(file) for file in os.scandir(os.path.join(root_path, type))]
        images += curr_data
        labels += [0 if type == "benign" else 1 for _ in range(len(curr_data))]
    return images, labels

### Construct Datasets
### We use monai (based on PyTorch) to load datasets
def construct_datasets(images,
                       labels,
                       pinmemory,
                       is_train=True,
                       batch_size=4,
                       rand_flip=False):
    """
    :param images: list of image file names
    :param labels: list of labels (0: benign, 1: malignant)
    :param pinmemory: bool, torch.cuda.is_available()
    :param is_train: bool, True if using train else False
    :param batch_size: int, batch size
    :param rand_flip: float
    :return:
    """
    # Define transforms
    if is_train:
        # optional random flipping for training data
        if rand_flip:
            transforms = Compose([ScaleIntensity(),
                                  EnsureChannelFirst(),
                                  ResizeWithPadOrCrop((460, 700)),
                                  RandFlip(prob=0.5, spatial_axis=None),])
        else:
            transforms = Compose([ScaleIntensity(),
                                  EnsureChannelFirst(),
                                  ResizeWithPadOrCrop((460, 700))])
    else:
        # test data transform
        transforms = Compose([ScaleIntensity(),
                              EnsureChannelFirst(),
                              ResizeWithPadOrCrop((460, 700))])

    # create a data loader
    ds = ImageDataset(image_files=images, labels=labels, transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pinmemory)

    return ds, loader



