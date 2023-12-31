import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import PySimpleGUI as sg

from utils import *

"""
DEFINE GUI
"""

img = sg.Image(key='image', expand_x=True, expand_y=True)

layout = [
    [sg.Titlebar('Breast Cancer Histopathological Image Classification and Cell Property Measurement')],

    [sg.Text('Input Image'), sg.FileBrowse(key='input_image')],
    [sg.Text('Output Image'), sg.FolderBrowse(key='output_path')],

    ####### Need Further Modify 
    [sg.Text('Guassian Sigma')], 
    [sg.Input(key='sigma')],

    [sg.Button('Classify'), sg.Button('Exit')],

    [sg.Text("", font=50, key='below')],

    [sg.Text('', font=30, key='results')],
    [sg.Text('Cell Number: ', key='cell_num')],

    [sg.Text('Cell Population Density: ', key='cell_population_density')],
    [sg.Text('Nuclear Proportion: ', key='nuclear_cytoplasmic_ratio')],

    [sg.Text('Nuclear Mean Area: ', key='nuclear_mean_size')],
    [sg.Text('Nuclear Mean Intensity: ', key='nuclear_mean_intensity')],

    [sg.Text('Nuclear Shape Mean Irregularity: ', key='nuclear_shape_mean_irregularity')],
    [sg.Text('Nuclear Shape Std Irregularity: ', key='nuclear_shape_std_irregularity')],

    [sg.Text('Tumor Type Classification: ', key='class')],

    [sg.Text("", font=30)],

    # [sg.Image(key='image', expand_x=True, expand_y=True)]
    [sg.Column([[img]], justification='center')]
]

new_layout  =  []

window = sg.Window('Classfier', layout=layout,
                   size=(800, 1000))

from multiprocessing.spawn import freeze_support

if __name__ == "__main__":
    freeze_support()

    # window.extend_layout(window['below'], new_layout)
    # window.refresh()

    while(True):
        event, values = window.read()

        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break

        if event == 'Classify':
            if values['input_image'] == '':
                sg.popup('Please choose an image first!')
            elif values['output_path'] == '':
                sg.popup('Please specify a name for the output image first!')
                
            else:
                window['results'].update("Results Loading...")
                # Diaplay cell properties 
                properties, img_with_border = main_func(img_path = values['input_image'], sigma=int(values['sigma']))

                window['results'].update("Analysis Finished: ")
                for property in properties.keys():
                    value = properties[property]
                    window[property].update(" ".join([i.capitalize() for i in property.split("_")]) + ": " + str(value))
            
                label = model_inference(img_pth=values['input_image'],
                        model_pth="./models/best_metric_model_classification_Dense121_9560.pth").cpu()
                label = np.array(label)[0]

                window['class'].update('Tumor Type Classification: ' + ("malignant" if label else "benign"))
           
                io.imsave(values['output_path']+'/output.png', img_with_border)
                window['image'].update(values['output_path']+'/output.png')
