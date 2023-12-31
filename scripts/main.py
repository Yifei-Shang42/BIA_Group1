import skimage.io as io
import PySimpleGUI as sg

from utils import *

##### Modified Based on PySimpleGUI demo Demo_Design_Pattern_Multiple_Windows.py
##### Website: https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Design_Pattern_Multiple_Windows.py

"""
DEFINE GUI
"""

def make_win1():
    """
    Define Initial GUI Window 

    :return: a window 
    """
    # define the window layout 
    layout = [
        [sg.Titlebar('Breast Cancer Histopathological Image Classification and Cell Property Measurement')],

        [sg.Text('Input Image'), sg.FileBrowse(key='input_image')],
        [sg.Text('Output Image'), sg.FolderBrowse(key='output_path')],

        [sg.Text('Guassian Sigma: '), 
         sg.Combo(values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key='sigma',
                default_value=7, readonly=True)], 

        [sg.Button('Classify'), sg.Button('Exit')],

        [sg.Text("", font=50, key='below')],
]
    return sg.Window('Classify', layout, location=(800,600), finalize=True)  # create the window 


def make_win2():
    """
    Define Results Displaying Window

    :return: a window 
    """
    # define the window layout 
    img = sg.Image(key='image', expand_x=True, expand_y=True)
    layout  =  [

        [sg.Text('Results Loading...It may take a minute', font=30, key='results')], 
        [sg.Text('Cell Number: ', key='cell_num')],

        [sg.Text('Cell Population Density: ', key='cell_population_density')], 
        [sg.Text('Nuclear Proportion: ', key='nuclear_cytoplasmic_ratio')],
        
        [sg.Text('Nuclear Mean Area: ', key='nuclear_mean_size')],
        [sg.Text('Nuclear Mean Intensity: ', key='nuclear_mean_intensity')],

        [sg.Text('Nuclear Shape Mean Irregularity: ', key='nuclear_shape_mean_irregularity')], 
        [sg.Text('Nuclear Shape Std Irregularity: ', key='nuclear_shape_std_irregularity')],

        [sg.Text('Tumor Type Classification: ', key='class')], 

        [sg.Text("", font=30)],

        [sg.Column([[img]], justification='center')],

        [sg.Button('Exit')]
    ]
    return sg.Window('Results', layout, finalize=True)  # create the window 


def main():
    """
    The Main Function to Be Called to Initialize GUI 

    :return: a series of window events 
    """
    window1, window2 = make_win1(), None  # start with window1 created 

    while True:  # create with event loop 
        window, event, values = sg.read_all_windows()
        if event == sg.WIN_CLOSED or event == 'Exit':
            window.close()

            if window == window2:  # if window2 closed, still keep window1 
                window2 = None
            elif window == window1:  # if window2 closed, program finishes 
                break

        elif event == 'Classify' and not window2:  # define the most important event 
            
            if values['input_image'] == '':  # check with input path 
                sg.popup('Please choose an image first!')
            elif values['output_path'] == '':  # check with output path 
                sg.popup('Please specify a name for the output image first!')
                
            else:
                window2 = make_win2()  # pop out result analyzing window 
                window2.refresh()
                
                # obtain and display cell properties 
                properties, img_with_border = main_func(img_path = values['input_image'], sigma=int(values['sigma']))

                window2['results'].update("Results Loading...Analysis Finished: ")
                for property in properties.keys():
                    value = properties[property]
                    window2[property].update(" ".join([i.capitalize() for i in property.split("_")]) + ": " + str(value))

                # obtain and display tumor type 
                label = model_inference(img_pth=values['input_image'],
                        model_pth="./models/best_metric_model_classification_Dense121_9560.pth").cpu()
                label = np.array(label)[0]

                window2['class'].update('Tumor Type Classification: ' + ("malignant" if label else "benign"))

                # display the image with nuclei edge detected 
                io.imsave(values['output_path']+'/output.png', img_with_border)
                window2['image'].update(values['output_path']+'/output.png')
                
    window.close()

# to avoid errors raised from multi-threading
if __name__ == '__main__':
    main()
