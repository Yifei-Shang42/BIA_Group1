import matplotlib.pyplot as plt
import numpy as np
import skimage.io as im
import sklearn.cluster
from scipy import ndimage as ndi
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from utils import *

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
ALTERNATIVE
"""
# blur = cv2.GaussianBlur(img, (3,3), 0)
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# # Perform connected component labeling
# n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)


