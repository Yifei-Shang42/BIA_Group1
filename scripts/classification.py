"""
BIA4 Group 1: Breast Cancer
"""

"""
IMPORT
"""
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
import numpy as np
import skimage.io as im
from torch.utils.tensorboard import SummaryWriter
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

from scripts.utils import *

pinmemory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
LOAD DATA
"""
train_img, train_labels = load_data_labels(root_path="./data/train")
test_img, test_labels = load_data_labels(root_path="./data/test")


"""
CONSTRUCT DATASETS
"""
train_ds, train_loader = construct_datasets(images=train_img,
                                            labels=train_labels,
                                            batch_size=4,
                                            pinmemory=pinmemory,
                                            use_kmeans=True)


test_ds, test_loader = construct_datasets(images=test_img,
                                          labels=test_labels,
                                          batch_size=4,
                                          pinmemory=pinmemory,
                                          use_kmeans=True)


"""
MODEL & LOSS & OPTIMIZER
"""
# Create DenseNet121, CrossEntropyLoss and Adam optimizer
# model = monai.networks.nets.SEResNet101(spatial_dims=2, in_channels=3, num_classes=2).to(device)
# model = monai.networks.nets.SEResNet152(spatial_dims=2, in_channels=3, num_classes=2).to(device)
model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=4, out_channels=2).to(device)
# model = monai.networks.nets.ViT(in_channels=3,
#                                 img_size=(700, 460),
#                                 patch_size=(16, 16),
#                                 classification=True,
#                                 num_classes=2,
#                                 spatial_dims=2).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)


"""
TRAINING
"""
## Training set-up
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
max_epochs = 100
save_name = "./models/best_metric_model_classification_DenseNet121_kmeans.pth"
initial_start_time = time.time()
start_time = initial_start_time
model_name = "DenseNet121"

## Main Training
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

        #############QUICK AND DIRTY, MODIFY INTO FUNCTION LATER
        if model_name == "ViT":
            outputs = model(inputs)[0]
            # print(outputs)
        else:
            outputs = model(inputs)
            # print(outputs)

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
        val_loss_sum = 0################
        num_correct = 0.0
        metric_count = 0
        for val_data in test_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            with torch.no_grad():
                if model_name == "ViT":
                    val_outputs = model(val_images)[0]
                else:
                    val_outputs = model(val_images)

                val_loss = loss_function(val_outputs, val_labels)#########
                val_loss_sum += val_loss.item()#############

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


"""
CAM
"""
# Load trained model
model = load_trained_model(model_pth="./models/best_metric_model_classification_Dense121_9578.pth",
                           model_class=monai.networks.nets.DenseNet121,
                           device=device)

# Visualize
show_cam_of_img(model,
                img_pth=train_img[22],
                device=device)


"""
INFERENCE
"""
train_ds, train_loader = construct_datasets(images=train_img,
                                            labels=train_labels,
                                            is_train=True,
                                            batch_size=1,
                                            pinmemory=pinmemory)
wrong_images = []
wrong_labels = []
model.eval()
num_correct = 0.0
metric_count = 0
for val_data in train_loader:
    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
    with torch.no_grad():
        val_outputs = model(val_images)
        value = torch.eq(val_outputs.argmax(dim=-1), val_labels)
        metric_count += len(value)
        num_correct += value.sum().item()
        if value.sum().item() != val_images.shape[0]:
            wrong_images.append(val_images)
            wrong_labels.append(val_labels)
metric = num_correct / metric_count

# viz image
for img in wrong_images[5:]:
    im.imshow(np.array(img.cpu())[0].swapaxes(0, -1))
    im.show()
    break

