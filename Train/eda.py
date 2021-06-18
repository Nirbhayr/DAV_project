# -*- coding: utf-8 -*-
"""
Created on Mon May 17 03:23:41 2021

@author: Nirbhay Raghav

"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
from prettytable import PrettyTable
import pickle
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import PIL
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

#%% Loading data

def load_data():
    '''
    Loads data into a pandas Dataframe object and return them as dataframe object
    Parameters
    ----------
    None

    Returns
    -------
    type : pandas DataFrame object
    returns 2 dataframes each for train and test data from a fixed directory
    
    '''
    
    path = 'G:\\Data Analysis and visualization\\Project\\'
    train_path = 'G:\\Data Analysis and visualization\\Project\\train_images\\'
    test_path = 'G:\\Data Analysis and visualization\\Project\\test_images\\'
    
    # read data from the csv file to have labelled table for each image id
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv')) 
    
    # add a column called image path to the table for having file path of each image
    train['image_path'] = train['id_code'].map(lambda x: os.path.join(train_path,'{}.png'.format(x)))
    test['image_path'] = test['id_code'].map(lambda x: os.path.join(test_path,'{}.png'.format(x)))
    
    # adding a columm filename in the dataframe to have filename as a png extension to access
    train['file_name'] = train["id_code"].apply(lambda x: x + ".png")
    test['file_name'] = test["id_code"].apply(lambda x: x + ".png")
    
    #convert the numbers in lables to strings
    
    train['diagnosis'] = train['diagnosis'].astype(str)
    
    return train, test

df_train, df_test = load_data()
print(df_train.shape, df_test.shape)

#%% Plotting data distribution

def plot_class_dist(df):
    """
    Plots the class wise distribution of label;ed dataframe passed.

    Parameters
    ----------
    df : DataFrame
        Dataframe for the dataset whose distribution is to be plotted

    Returns
    -------
    type : numpy array
    returns a numpy array of counts of each class as a table
    
    """
    
    #using groupby to group the values by diagnosis
    df_group = df.groupby('diagnosis').agg('size').reset_index()
    df_group.columns = ['diagnosis', 'Counts']
    
    
    sns.barplot(x = 'Diagnosis', y = 'Counts', data = df_group)
    
    plt.title('Class disribution')
    plt.show()
    return df_group.value_counts()


dist = plot_class_dist(df_train)
print(dist)


#%% Visualization of Images
IMG_SIZE = 320
def visualize(df, color_scale):
    
    df = df.groupby('diagnosis',group_keys = False).apply(lambda df: df.sample(3))
    df = df.reset_index(drop = True)
    
    print(df.head())
    
    plt.rcParams["axes.grid"] = False
    
    for i in range(3):
        
        f, axarr = plt.subplots(1,5,figsize = (15,15))
        axarr[0].set_ylabel("Sample Data Points")
        
        df_temp = df[df.index.isin([i + (3*0), i + (3*1), i + (3*2), i + (3*3), i + (3*4)])]
        
        # print(df_temp)
        for j in range(5):
            
            axarr[j].imshow(Image.open(df_temp.image_path.iloc[j]).resize((IMG_SIZE,IMG_SIZE)), cmap = color_scale)
            axarr[j].set_xlabel('Class '+str(df_temp.diagnosis.iloc[j]))

        plt.show()

visualize(df_train, color_scale= 'gray')

#%%
df_train_train, df_train_valid = train_test_split(df_train, test_size = 0.2)

#%%
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def circle_crop(img, sigmaX = 30):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 
#%% 
def preprocess_image(file:list):
    
    for i in range(len(file)):
        
        input_filepath = os.path.join(path , '\\train_images_resized\\{}'.format(file[i]))
        output_filepath = path + '\\train_images_resized_and_processed\\{}'.format(file)
        print(input_filepath)
        img = cv2.imread(input_filepath)
        img = circle_crop(img) 
        cv2.imwrite(output_filepath, cv2.resize(img, (IMG_SIZE,IMG_SIZE)))


def multiprocess_image_processor(process:int, imgs:list):
    """
    Inputs:
        process: (int) number of process to run
        imgs:(list) list of images
    """
    print(f'MESSAGE: Running {process} process')
    proc = ThreadPool(process)
    preprocess_image(imgs)

multiprocess_image_processor(12, list(df_train_train.id_code.values)) # since id codes are same for train_train images and resize
#train images and anyhow the inputfilepath will take care to process resized images only

