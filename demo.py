import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from os import listdir

## currently, we use 3 models
modelNames = ['ResNet_model_03', 'Inception_model_03', 'Xception_model_03']

image_size = (256, 256, 3)

category_list = ['hamburger', 'pizza', 'ramen', 'sushi']

def checkGPU():
    
    physical_devices = tf.config.list_physical_devices('GPU')

    print(physical_devices)

    if len(physical_devices) >= 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

def checkImage(imagePath):
    # "check whether the input is an valid image format"
    imagePath = imagePath.lower()
    if imagePath.endswith(".jpg") or imagePath.endswith(".jpeg") or imagePath.endswith(".png"):
        return True
    else:
        return False
    
    
def loadModel():
    models = []
    
    for name in modelNames:
        model = keras.models.load_model(name)
        models.append(model)
    
    return models

def predictResult(input_model, input_image_path):
    '''
    perform prediction, get predict category and probability
    '''
    image = keras.preprocessing.image.load_img(path=input_image_path, target_size = image_size )
    image_tensor = keras.preprocessing.image.img_to_array(image)
    predictProbabilities = input_model(image_tensor, training=False).numpy()

    #print(predictProbabilities)
    predict_index = predictProbabilities[0].argmax(axis=-1)
    #print("predict index is ", predict_index)
    predict_category = category_list[predict_index]
    probability = predictProbabilities[0][predict_index]

    return predict_category, probability

def processVottingData(categories, probabilities):
    counter = 0
    category = None
    for i in categories:
        curr_frequency = categories.count(i)
        if (curr_frequency) > counter:
            counter = curr_frequency
            category = i
    
    if counter == 1:
        # if the max counter is 1, then we select the category which has max prediction probabilities. 
        index = probabilities.index(max(probabilities))
        return categories[index]
    else:
        # else return the category with the most frequency
        return category    

def modelVotting(models, image):
    ## for each model, get the prediction result. Then make majority votting
    temp_category = []
    temp_probability = []
    
    for model in models:
        predict_category, predict_probability = predictResult(model, image)
        temp_category.append(predict_category)
        temp_probability.append(predict_probability)
        
    ## perform majority votting
    votting_category = processVottingData(temp_category, temp_probability)
    return votting_category     ## return votting result
    
def run(input_path):
    
    checkGPU()      ## check whether GPU is available or not
    
    image_list = []
    
    ## check whether it is image path or folder
    if os.path.isdir(input_path):   ## if it is folder, then add all the images into list
        file_list = listdir(input_path)
        for file in file_list:
            if checkImage(file):
                image_list.append(input_path + "/" + file)
            else:
                print("current code only accepts image file with suffix .jpg, .jpeg and .png, {} is not accepted".format(file))
    else:
        if checkImage(input_path):
            image_list.append(input_path)
        else:
            print("current code accepts image file with suffix .jpg, .jpeg or .png,  {} is not accepted".format(input_path))
    
    if len(image_list) > 0:
        ## check whether it is image or folder
        print("Start to load CNN models......")
        models = loadModel()
        print("Finished loading CNN models, start to predict and vote......")
        
        for image in image_list:
            votting_result = modelVotting(models, image)
            fileName = image.split("/")[-1]
            print("The predict result for image {} is {}".format(fileName, votting_result))
        
    print("Demo finished")
    
    return
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a parser')
    parser.add_argument('--input', type=str, default="demo_images",
                        help='input for prediction, it can be an image or a folder')
    args = parser.parse_args()
    run(args.input)
    