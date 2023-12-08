"""
Function Storage
"""

### Import
import datetime
import os
import time
import scipy.spatial
import scipy.signal
import torch
import monai
import numpy as np
import pandas as pd
import skimage.io as im
from scipy import ndimage as ndi
import sklearn.cluster
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    ResizeWithPadOrCrop
)


### Load Data
def load_data_labels(root_path="./data/train"):
    """
    Load Data Path and Labels 

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
                       batch_size=4):
    """
    Construct Datasets and Prepare for Training 

    :param images: list of image file names
    :param labels: list of labels (0: benign, 1: malignant)
    :param pinmemory: bool, torch.cuda.is_available()
    :param batch_size: int, batch size
    :param use_kmeans: bool, True if using KMeans
    :return: datasets
    """
    # Define transforms
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
    Load the Model from Defined Path 

    :param model_pth: str, path of stored model state dict
    :param model_class: a model Class, compatible with state dict
    :param device: str, "cpu" / "cuda"
    :return: loaded model
    """
    model = model_class(spatial_dims=2, in_channels=3, out_channels=2).to(device)
    state_dict = torch.load(model_pth, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


### Model Training
def model_train(max_epochs,
                model,
                optimizer,
                train_loader,
                train_ds,
                writer,
                device,
                loss_function,
                val_interval,
                save_name,
                test_loader):
    """
    Train the Model 

    :param max_epochs: int, maximum epochs
    :param model: initialized model used for training
    :param optimizer: initialized optimizer used for training
    :param train_loader: constructed train data loader
    :param train_ds: constructed train dataset
    :param writer: summary history writer
    :param device: GPU/CPU
    :param loss_function: initialized loss function for training
    :param val_interval: int, validation frequency
    :param save_name: str, model save name
    :param test_loader: constructed test data loader
    :return: None
    """
    # set up
    best_metric = -1
    best_metric_epoch = -1
    initial_start_time = time.time()
    start_time = initial_start_time
    epoch_loss_values = []
    metric_values = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        ## Training step
        for batch_data in train_loader:
            step += 1
            inputs, train_labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, train_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            if step % 50 == 0:
                ## TIMER MODULE
                finish_time = time.time()
                time_diff = finish_time - start_time
                est_total_time_per_epoch = (len(train_loader) / 50) * time_diff
                current_finish_ratio = (epoch * len(train_loader) + step) / (max_epochs * len(train_loader))
                seconds_to_finish = max_epochs * est_total_time_per_epoch * (1 - current_finish_ratio)
                start_time = finish_time

                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                print(f"Estimated Time Left: {str(datetime.timedelta(seconds=seconds_to_finish))}")

            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        ## TIMER MODULE
        finish_time

        ## Evaluation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss_sum = 0
            num_correct = 0.0
            metric_count = 0
            for val_data in test_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    val_loss = loss_function(val_outputs, val_labels)
                    val_loss_sum += val_loss.item()

                    value = torch.eq(val_outputs.argmax(dim=-1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), save_name)
                print("saved new best metric model")
            print(f"Current epoch: {epoch + 1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

### Model Inference
def model_inference(img_pth,
                    model_pth):
    """
    Predict with Trained Model 

    :param img_pth: image path to be classified
    :param model_pth: pretrained model path
    :return: model prediction of the image class
    """
    # detect GPU / CPU
    pinmemory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained model
    model = load_trained_model(model_pth=model_pth,
                               model_class=monai.networks.nets.DenseNet121,
                               device=device)
    model.eval()

    # construct dataset based on read image
    ds, loader = construct_datasets(images=[img_pth],
                                    labels=[-1],
                                    batch_size=1,
                                    pinmemory=pinmemory)

    # inference
    for data in loader:
        img, label = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(img)
            value = outputs.argmax(dim=-1)
    return value


### KMeans & Watershed
def kmeans_watershed_nuclei_seg(img, sigma: int = 7, 
                                footprint: np.array = np.ones((5, 5)), 
                                min_distance: int=10, 
                                min_size: int=300):
    """
    Do KMeans and Watershed Segmentation
    Calculate Cell Population and Nuclea-Cytoplasmic Ratio 

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
    cell_population_density = round(len(labels[(labels == cyto_label) | (labels == nuclei_label)].ravel())/len(img.ravel())*100, ndigits=3)
    nuclear_cytoplasmic_ratio = round(len(labels[labels == nuclei_label].ravel())/len(labels[labels == cyto_label].ravel())*100, ndigits=3)

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

    return (segmented, cell_population_density, nuclear_cytoplasmic_ratio) 


### Cell Property Measurement 
def cell_property(img, seg):
    """
    Calculate Cell Number and Nuclear Size 

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
    nuclear_mean_size = round(np.mean(data.area), ndigits=3)
    nuclear_mean_intensity = round(np.mean(data.mean_intensity), ndigits=3)

    return (cell_num, nuclear_mean_size, nuclear_mean_intensity, data)


### Get Border from Instance Segmentation
def border_cell_from_ins_map(ins_map):
    """
    Calculated Nuclear Shape Irregularity 

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
def main_func(img_path, sigma: int = 5):
    """
    Main Function to Be Called for Analysis 

    :param img_path: input image path
    :return: dictionary of cell properties, img with border marked
    """
    # read data
    img = im.imread(img_path)
    properties = {}

    # KMEANS & WATERSHED
    res, cell_population_density, nuclear_cytoplasmic_ratio = kmeans_watershed_nuclei_seg(img, sigma=sigma)

    # CELL PROPERTY MEASUREMENT
    cell_num, nuclear_mean_size, nuclear_mean_intensity, data = cell_property(img, res)

    # BORDER & SHAPE IRREGULARITY
    plot_df, nuclear_shape_mean_irregularity, nuclear_shape_std_irregularity = border_cell_from_ins_map(res)
    img_with_border = img.copy()
    img_with_border[plot_df != 0, :] = 0

    # Summarize Results
    properties["cell_population_density"] = cell_population_density
    properties["nuclear_cytoplasmic_ratio"] = nuclear_cytoplasmic_ratio
    properties["cell_num"] = cell_num
    properties["nuclear_mean_size"] = nuclear_mean_size
    properties["nuclear_mean_intensity"] = nuclear_mean_intensity
    properties["nuclear_shape_mean_irregularity"] = nuclear_shape_mean_irregularity
    properties["nuclear_shape_std_irregularity"] = nuclear_shape_std_irregularity

    return properties, img_with_border

