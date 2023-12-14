# BIA4 Group 1: Breast Cancer
 
## Introduction 
This software can do **breast cancer cell segmentation** and **tumor type classification**. 

Based on the segmentation results, cell properties such as `cell population density`, `nuclear-cytoplasmic ratio`, `nuclear size` and `nuclear shape irregularity` were measured, providing more information to clinical experts to aid with microscopic biopsy diagnosis. 

## Documentation 

See `./scripts/documentation.ipynb` for more information. 

## Install
Download the package from this GitHub repository. The data used to build this classifier are included in the `data` directory. 

## Dependencies 
Note that we use Python 3.10 for software development. The majority of package versions are the latest. However, if you are using older versions, it might also work. 

| Package  | Package | Package |
|:--------:|:-------:|:-------:|
|matplotlib|  monai  |numpy|
|pandas|PySimpleGUI|scipy|
|skimage|sklearn|torch|

## Input 
Breast tumor microscopic biopsy images with `700x460` size. 

## Usage

You can run the following code to initialize the GUI of our software. 

``` bash
python3 ./scripts/main.py
```

## History Visualization

You can visualize the training history of our pretrained model using Tensorboard by running the following script
``` bash
tensorboard --logdir .\runs\DenseNet121_history
```

## References 
Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016). A Dataset for Breast Cancer Histopathological Image Classification. IEEE transactions on bio-medical engineering, 63(7), 1455–1462. https://doi.org/10.1109/TBME.2015.2496264

