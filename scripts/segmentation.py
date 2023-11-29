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

from utils import *

from multiprocessing.spawn import freeze_support

if __name__ == "__main__":
    freeze_support()

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

    res, cell_population_density, nuclear_cytoplasmic_ratio = kmeans_watershed_nuclei_seg(img, sigma=5)
    print(cell_population_density, nuclear_cytoplasmic_ratio)

    plt.imshow(res, cmap="viridis")
    plt.show()


    """
    CELL PROPERTY MEASUREMENT 
    """
    cell_num, nuclear_mean_size, nuclear_mean_intensity, data = cell_property(img, res)
    print(cell_num, nuclear_mean_size, nuclear_mean_intensity)


    """
    BORDER & SHAPE IRREGULARITY
    """
    plot_df, nuclear_shape_mean_irregularity, nuclear_shape_std_irregularity = border_cell_from_ins_map(res)
    print(nuclear_shape_mean_irregularity, nuclear_shape_std_irregularity)

    # plot img with border
    img_with_border = img.copy()
    img_with_border[plot_df != 0, :] = 0

    plt.imshow(img_with_border)
    plt.show()


    """
    MODEL INFERENCE
    """


    label = model_inference(img_pth=train_img[0],
                            model_pth="./models/best_metric_model_classification_Dense121_9560.pth").cpu()
    print(np.array(label)[0])


    """
    MAIN FUNC
    """
    properties, img_with_border = main_func(train_img[1], sigma=5)
    print(properties)

