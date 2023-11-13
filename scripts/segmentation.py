import matplotlib.pyplot as plt
import numpy as np
import skimage.io as im
import sklearn.cluster
from scipy import ndimage as ndi
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import scipy.spatial
import scipy.signal

from scripts.utils import *

"""
LOAD DATA
"""
train_img, train_labels = load_data_labels(root_path="./data/train")
test_img, test_labels = load_data_labels(root_path="./data/test")


"""
KMEANS & WATERSHED
"""
img = im.imread(train_img[1])
im.imshow(img)
im.show()

res, cell_density, nuclear_proportion = kmeans_watershed_nuclei_seg(img, sigma=5)
print(cell_density, nuclear_proportion)

plt.imshow(res, cmap="viridis")
plt.show()


"""
CELL PROPERTY MEASUREMENT 
"""
cell_num, cell_mean_area, cell_mean_intensity, data = cell_property(img, res)
print(cell_num, cell_mean_area, cell_mean_intensity)


"""
BORDER & SHAPE IRREGULARITY
"""
plot_df, mean_irregularity, std_irregularity = border_cell_from_ins_map(res)
print(mean_irregularity, std_irregularity)

# plot img with border
img_with_border = img.copy()
img_with_border[plot_df != 0, :] = 0

plt.imshow(img_with_border)
plt.show()


"""
MAIN FUNC
"""
properties, img_with_border = main_func(train_img[1])
print(properties)

