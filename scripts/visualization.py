import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import PySimpleGUI as sg

from scripts.utils import *


"""
DEFINE GUI
"""
layout = [
    [sg.Titlebar('Breast Cancer Histopathological Image Classification and Cell Property Measurement')],
    [sg.Text('Input Image'), sg.FileBrowse(key='input_image')],
    [sg.Text('Output Image'), sg.FolderBrowse(key='output_path')],
    [sg.Text('Guassian Sigma'), sg.Input('', enable_events=True, key='sigma')],
    [sg.Text('ColorMap'), 
     sg.Combo(values=['gray', 'viridis', 'Greens', 'Blues', 'Reds'], key='cmap', default_value='gray', readonly=True)], 
    [sg.Button('Classify'), sg.Button('Exit')],
    [sg.Text("", font=50)],
    [sg.Text('Results: ', font=30)], 
    [sg.Text('Classification: ', key='class')], 
    [sg.Text('Cell Density: ', key='cell_density')], 
    [sg.Text('Nuclear Proportion: ', key='nuclear_proportion')],
    [sg.Text('Cell Number: ', key='cell_num')],
    [sg.Text('Cell Mean Area: ', key='cell_mean_area')],
    [sg.Text('Cell Mean Intensity: ', key='cell_mean_intensity')],
    [sg.Text('Cell Mean Irregularity: ', key='mean_irregularity')], 
    [sg.Text('Cell Std Irregularity: ', key='std_irregularity')],
    [sg.Image(key='image')]
]

window = sg.Window('Classfier', layout=layout, size=(500, 500))


while(True):
    event, values = window.read()

    if event == 'Exit' or event == sg.WIN_CLOSED:
        window.close()
        break

    if event == 'Classify':
        if values['input_image'] == '':
            sg.popup('Please choose an image first!')
        elif values['output_image'] == '':
            sg.popup('Please specify a name for the output image first!')
            
        else:
            # Diaplay cell properties 
            properties, img_with_border = main_func(img_path = values['input_image'], sigma=int(values['sigma']))
            
            cell_density = properties["cell_density"]
            window['cell_density'].update("Cell Density: " + cell_density)

            nuclear_proportion = properties["nuclear_proportion"]
            window['nuclear_proportion'].update('Nuclear Proportion: ' + nuclear_proportion)

            cell_num = properties["cell_num"]
            window['cell_num'].update('Cell Number: ' + cell_num)

            cell_mean_area = properties["cell_mean_area"]
            window['cell_mean_area'].update('Cell Mean Area: ' + cell_mean_area)

            cell_mean_intensity = properties["cell_mean_intensity"] 
            window['cell_mean_intensity'].update('Cell Mean Intensity: ' + cell_mean_intensity)

            mean_irregularity = properties["mean_irregularity"]
            window['mean_irregularity'].update('Cell Mean Irregularity: ' + mean_irregularity)

            std_irregularity = properties["std_irregularity"]
            window['std_irregularity'].update('Cell Std Irregularity: ' + std_irregularity)


            label = model_inference(img_pth=values['input_image'],
                    model_pth="./models/best_metric_model_classification_Dense121_9560.pth").cpu()
            label = np.array(label)[0]
            window['class'].update('Classification: ' + label)

            
            # Diaplay and Save the image 
            # plt.imshow(img_with_border, cmap=values['cmap'])
            # plt.show()
            
            io.imsave(values['output_path']+'/output.png', img_with_border, cmap=values['cmap'])
            window['image'].update(values['output_path']+'/output.png')
