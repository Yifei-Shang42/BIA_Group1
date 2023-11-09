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
from scipy import ndimage as ndi
import sklearn.cluster
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from monai.config import print_config
from monai.visualize import CAM
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
                                  ResizeWithPadOrCrop((700, 460)),
                                  RandFlip(prob=0.5, spatial_axis=None),])
        else:
            transforms = Compose([ScaleIntensity(),
                                  EnsureChannelFirst(),
                                  ResizeWithPadOrCrop((700, 460))
            ])
    else:
        # test data transform
        transforms = Compose([ScaleIntensity(),
                              EnsureChannelFirst(),
                              ResizeWithPadOrCrop((700, 460))])

    # create a data loader
    ds = ImageDataset(image_files=images, labels=labels, transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pinmemory)

    return ds, loader

### Load Trained Model
def load_trained_model(model_pth,
                       model_class,
                       device):
    """
    :param model_pth: str, path of stored model state dict
    :param model_class: a model Class, compatible with state dict
    :param device: str, "cpu" / "cuda"
    :return: loaded model
    """
    model = model_class(spatial_dims=2, in_channels=3, out_channels=2).to(device)
    state_dict = torch.load(model_pth)
    model.load_state_dict(state_dict)
    return model

### Show CAM with img
def show_cam_of_img(model,
                    img_pth,
                    device):
    """
    :param model: loaded model
    :param img_pth: str, image path
    :param device: str, "cpu" / "cuda"
    :return: visualizes cam with img
    """
    # get cam of model
    cam = CAM(nn_module=model,
              target_layers="class_layers.relu",
              fc_layers="class_layers.out")

    # load img & preprocess
    img = im.imread(img_pth)
    img_cf = torch.unsqueeze(torch.reshape(torch.tensor(img).to(torch.float32),
                                           (3, 460, 700)), 0).to("cuda")

    # gen acti map
    result = cam(x=img_cf)
    curr_acti = np.array(result[0, 0, :, :].cpu())

    # visualization
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(curr_acti, cmap="gray")
    plt.show()
    return

### Model Inference
def model_inference(model,
                    train_ds,
                    device):
    model.eval()


### KMeans & Watershed
def kmeans_watershed_nuclei_seg(img):
    """
    :param img: numpy array, image to be segmented
    :return: segmentation of single nuclei
    """
    # KMeans
    img_flat = img.transpose(2, 0, 1).reshape(3, -1).T
    model = sklearn.cluster.KMeans(n_clusters=3).fit(img_flat)
    labels = model.labels_.reshape(460, 700)

    # determine nuclei
    mean1 = np.mean(img[labels == 0])
    mean2 = np.mean(img[labels == 1])
    mean3 = np.mean(img[labels == 2])
    nuclei_label = np.argmin([mean1, mean2, mean3])
    labels[labels != nuclei_label] = 10
    labels[labels == nuclei_label] = 1
    labels[labels == 10] = 0

    # watershed
    distance = ndi.distance_transform_edt(labels)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=labels)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    res = watershed(-distance, markers, mask=labels)

    return res
