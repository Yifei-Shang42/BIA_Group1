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
import scipy.spatial
import scipy.signal
import torch
import monai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as im
from torch.utils.tensorboard import SummaryWriter
from scipy import ndimage as ndi
import sklearn.cluster
import cv2
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian
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
def kmeans_watershed_nuclei_seg(img, sigma: int = 7, 
                                footprint: np.array = np.ones((5, 5)), 
                                min_distance: int=10, 
                                min_size: int=300):
    """
    :param img: numpy array, image to be segmented
    :param sigma: int, sigma of Guassian Filter, default value is 7
    :param footprint: numpy array, the footprint in peak_local_max calculation 
    :param min_distance: int, the min distance in peak_local_max calculation 
    :param min_size: int, the min size in small objects removement 
    :return: segmentation of single nuclei and general cell properties 
    """
    # pre-processing 
    img = gaussian(img, sigma=sigma, channel_axis=-1)

    # KMeans
    img_flat = img.transpose(2, 0, 1).reshape(3, -1).T
    model = sklearn.cluster.KMeans(n_clusters=3).fit(img_flat)
    labels = model.labels_.reshape(460, 700)

    # determine nuclei and cytoplasm 
    mean1 = np.mean(img[labels == 0])
    mean2 = np.mean(img[labels == 1])
    mean3 = np.mean(img[labels == 2])
    nuclei_label = np.argmin([mean1, mean2, mean3])
    blank_label = np.argmax([mean1, mean2, mean3])
    cyto_label = set([0, 1, 2])
    cyto_label.discard(nuclei_label); cyto_label.discard(blank_label)
    cyto_label = cyto_label.pop()

    # measure cell density and nuclear proportion 
    cell_density = round(len(labels[(labels == cyto_label) | (labels == nuclei_label)].ravel())/len(img.ravel())*100, ndigits=3)
    nuclear_proportion = round(len(labels[labels == nuclei_label].ravel())/len(labels[labels == cyto_label].ravel())*100, ndigits=3)

    # watershed
    labels[labels != nuclei_label] = 10
    labels[labels == nuclei_label] = 1
    labels[labels == 10] = 0
    distance = ndi.distance_transform_edt(labels)
    coords = peak_local_max(distance, footprint=footprint, min_distance=min_distance, labels=labels)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    res = watershed(-distance, markers, mask=labels)

    # instance filtering 
    segmented = remove_small_objects(res, min_size=min_size)

    return (segmented, cell_density, nuclear_proportion) 


### Cell Property Measurement 
def cell_property(img, seg):
    """
    :param img: numpy array, orignal image 
    :param seg: numpy array, segmented img 
    :return: properties of single cell region 
    """

    # get region properties 
    props = regionprops(seg, intensity_image=img)
    tbl = regionprops_table(seg, intensity_image=img, 
                            properties=['label', 'area', 'area_bbox', 'bbox', 'intensity_mean'])
    tbl = pd.DataFrame(tbl)
    axis_row = [tbl.loc[i, 'bbox-2'] - tbl.loc[i, 'bbox-0'] for i in tbl.index]
    axis_col = [tbl.loc[i, 'bbox-3'] - tbl.loc[i, 'bbox-1'] for i in tbl.index]
    mean_intensity = [np.mean(tbl.loc[i, ['intensity_mean-0', 'intensity_mean-1', 
                                          'intensity_mean-2']]) for i in tbl.index]
    data = {'label': tbl.label, 'area': tbl.area, 'area_bbox': tbl.area_bbox, 
            'axis_row': axis_row, 'axis_col': axis_col, 
            'mean_intensity': mean_intensity}
    data = pd.DataFrame(data=data)

    # calculate cell properties 
    cell_num = data.shape[0]
    cell_mean_area = round(np.mean(data.area), ndigits=3)
    cell_mean_intensity = round(np.mean(data.mean_intensity), ndigits=3)

    return (cell_num, cell_mean_area, cell_mean_intensity, data)


### Get Border from Instance Segmentation
def border_cell_from_ins_map(ins_map):
    """
    :param ins_map: water shed segmentation res
    :return: cell borders for plot, mean & std of cell shape irregularity
    """
    # init res df
    roundness_arr = []
    plot_df = np.zeros(ins_map.shape)

    # get border for each cell: remove bgd with [1:]
    for cell in np.unique(ins_map)[1:]:
        # isolate this one cell
        curr_map = ins_map.copy()
        curr_map[curr_map != cell] = 0

        # calculate center of mass
        curr_center = ndi.center_of_mass(curr_map)

        # decide if each pixel is border
        kernel = np.array([[1,1,1],
                           [1,1,1],
                           [1,1,1]])

        # detect inner-vs-border using convolution
        conv_res = scipy.signal.convolve2d(curr_map, kernel,
                                           boundary="fill",
                                           fillvalue=0,
                                           mode="same")
        curr_map[conv_res == cell * 9] = 0
        plot_df += curr_map

        # calculate distance to center
        distances = []
        for coord in zip(*np.nonzero(curr_map)):
            curr_dis = scipy.spatial.distance.euclidean(coord, curr_center)
            distances.append(curr_dis)
        roundness = np.std(np.array(distances)/max(distances))
        roundness_arr.append(roundness)

    # shape irregularity properties
    mean_shape_irregularity = round(np.mean(np.array(roundness_arr)), 3)
    std_shape_irregularity = round(np.std(np.array(roundness_arr)), 3)

    return (plot_df, mean_shape_irregularity, std_shape_irregularity)


### Main Function
def main_func(img_path):
    """
    :param img_path: input image path
    :return: dictionary of cell properties, img with border marked
    """
    # read data
    img = im.imread(img_path)
    properties = {}

    # KMEANS & WATERSHED
    res, cell_density, nuclear_proportion = kmeans_watershed_nuclei_seg(img, sigma=5)

    # CELL PROPERTY MEASUREMENT
    cell_num, cell_mean_area, cell_mean_intensity, data = cell_property(img, res)

    # BORDER & SHAPE IRREGULARITY
    plot_df, mean_irregularity, std_irregularity = border_cell_from_ins_map(res)
    img_with_border = img.copy()
    img_with_border[plot_df != 0, :] = 0

    # Summarize Results
    properties["cell_density"] = cell_density
    properties["nuclear_proportion"] = nuclear_proportion
    properties["cell_num"] = cell_num
    properties["cell_mean_area"] = cell_mean_area
    properties["cell_mean_intensity"] = cell_mean_intensity
    properties["mean_irregularity"] = mean_irregularity
    properties["std_irregularity"] = std_irregularity

    plt.imshow(img_with_border)
    plt.show()

    return properties, img_with_border