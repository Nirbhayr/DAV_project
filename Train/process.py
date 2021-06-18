# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:22:58 2021

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

import multiprocessing as multi
from multiprocessing.pool import ThreadPool

from sklearn.model_selection import train_test_split

# 3 functions needed.
# load data as image, resize image, process image
IMG_SIZE = 512
path = 'G:\\Data Analysis and visualization\\Project\\'



#%% processing

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

 def train_preprocess_image(file):
     input_filepath = os.path.join(path,'train_train_images_resized','{}.png'.format(file))
     output_filepath = os.path.join(path, 'train_train_resized_processed','{}.png'.format(file))
    
     img = cv2.imread(input_filepath)
     img = circle_crop(img) 
     cv2.imwrite(output_filepath, cv2.resize(img, (IMG_SIZE,IMG_SIZE)))


def valid_preprocess_image(file):
    input_filepath = os.path.join(path,'test_images_resized','{}.png'.format(file))
    output_filepath = os.path.join(path, 'test_resized_processed','{}.png'.format(file))
    
    img = cv2.imread(input_filepath)
    img = circle_crop(img) 
    cv2.imwrite(output_filepath, cv2.resize(img, (IMG_SIZE,IMG_SIZE)))
    
def train_multiprocess_image_processor(process:int, imgs:list):
    """
    Inputs:
        process: (int) number of process to run
        imgs:(list) list of images
    """
    print(f'MESSAGE: Running {process} process')
    proc = ThreadPool(process).map(preprocess_image, imgs)

 def valid_multiprocess_image_processor(process:int, imgs:list):
     """
     Inputs:
         process: (int) number of process to run
         imgs:(list) list of images
     """
     print(f'MESSAGE: Running {process} process')
     proc = ThreadPool(process).map(valid_preprocess_image, imgs)

train_multiprocess_image_processor(12, list(df_train_train.id_code.values)) 

multiprocess_image_processor(12, list(df_test.id_code.values)) 









