"""
@author Krishna Karthik Gadiraju
# Write custom data generator for training and testing our network for crop classification 
# Label smoothing can be performed if needed, but is not enabled by default 
"""

import numpy as np
import os, glob
import pandas as pd
from random import shuffle
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical
#from visualize import visualize_time_series
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pandas as pd
from pathlib import Path
import sys

def one_hot_encode(class_names, datapoint):
    # class_names is a list
    # data point is a string
    class_names.sort()
    class_vector = np.zeros(len(class_names))
    class_index = class_names.index(datapoint)
    class_vector[class_index] = 1
    return(class_vector)

def get_all_files(data_path):
    """
    Extract all class names
    Input: data_path: str, path to the data folder containing all the classes 
    """
    classes_paths = glob.glob(os.path.join(data_path, "*"))
    classes_names = [os.path.basename(x) for x in classes_paths]
    classes_names.sort()
    all_data = {}
    for class_name in classes_names:
        curr_class_imgs = glob.glob(os.path.join(data_path, class_name, "*.csv"))
        for curr_img in curr_class_imgs:
            all_data[os.path.abspath(curr_img)] = int(class_name)
    return all_data     

def label_smooth(num_classes, epsilon, encoded_data_point):
    """
    Simple label smoothing 
    Input: num_classes: number of classes
           epsilon: epsilon value 
           encoded_data_point: one hot encoded data point
    """
    smoothed_point = encoded_data_point * [1. - epsilon] + (1. - encoded_data_point) * (epsilon/float(num_classes - 1.))
    return smoothed_point

def replace_nans(data):
    """
    Replace any nans with nearest values
    Source: https://stackoverflow.com/questions/9537543/replace-nans-in-numpy-array-with-closest-non-nan-value
    """
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return(data)


def preprocess_batches(ts_batch, class_names, num_classes, epsilon, mode):
    """
    Generate batches of time series data in format compatible with keras
    """
    return_x_ts = []
    return_y = []
    for ts_path in ts_batch:
        curr_class = int(os.path.basename(os.path.dirname(ts_path)))
        curr_ts = pd.read_csv(ts_path)
        if curr_ts.empty:
            print("{} seems to be empty: pls check ".format(ts_path))
            sys.exit(0)
        curr_ts_data = curr_ts['NDVI'].values
        curr_ts_data = replace_nans(curr_ts_data)
        curr_ts_data = curr_ts_data[:, np.newaxis]
        curr_ts_data = StandardScaler().fit_transform(curr_ts_data)
        curr_class_encoded = one_hot_encode(class_names, curr_class)  
        curr_class_label_smoothed = label_smooth(num_classes = num_classes, epsilon = epsilon, encoded_data_point = curr_class_encoded)
        return_y.append(curr_class_label_smoothed)
        return_x_ts.append(curr_ts_data)
    return_x_ts = np.array(return_x_ts, dtype=np.float32)
    assert len(return_x_ts) == len(return_y), "TrainingX time series and Y lengths mismatch!"
    return [return_x_ts, return_y]        

def preprocess_test_img(ts_path, class_names, num_classes, epsilon, mode):
    """
    Same as preprocess_batches but for test data. Separate function is more convenient.
    """
    return_x_ts = []
    return_y = []
    curr_class = int(os.path.basename(os.path.dirname(ts_path)))
    curr_ts = pd.read_csv(ts_path)
    if curr_ts.empty:
        print("{} seems to be empty: pls check ".format(ts_path))
        sys.exit(0)
    curr_ts_data = curr_ts['NDVI'].values
    curr_ts_data = replace_nans(curr_ts_data)
    curr_ts_data = curr_ts_data[:, np.newaxis]
    curr_ts_data = StandardScaler().fit_transform(curr_ts_data)
    return_y.append(curr_class)
    return_x_ts.append(curr_ts_data)
    return_x_ts = np.array(return_x_ts, dtype=np.float32)
    assert len(return_x_ts) == len(return_y), "TestX time series and Y lengths mismatch!"
    return [return_x_ts, return_y]        


    
def crop_generator(input_path, batch_size=32, mode="train", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True):	
    """
    Simple data generator that reads all images based on mode, picks up corresponding time series, returns entire list
    """
    data_path = os.path.join(input_path, mode)
    all_ts = glob.glob(os.path.join(data_path, "**/*.csv"))
    print("Found {} files for {}".format(len(all_ts), mode))
    if do_shuffle:
        shuffle(all_ts)
    curr_idx = 0
    while True:
        # initialize our batches of images and labels
        ts = []
        labels = []       
        if curr_idx > len(all_ts): # reset if you've parsed all data
            curr_idx = 0
        curr_batch = all_ts[curr_idx: (curr_idx + batch_size)]
        ts, labels = preprocess_batches(ts_batch= curr_batch, class_names = [0,1,2,3,4,5], num_classes = num_classes, epsilon = epsilon, mode=mode) 
        ts = np.array(ts)
        labels = np.array(labels)
        curr_idx += batch_size 
        yield (ts, labels)


def test_crop_generator(input_path, batch_size=1, mode="test", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True):
    """
    Simple data generator that reads all timeseries based on mode, picks up corresponding time series, returns entire list
    """
    data_path = os.path.join(input_path, mode)
    all_ts = glob.glob(os.path.join(data_path, "**/*.csv"))
    print("Found {} files for {}".format(len(all_ts), mode))
    if do_shuffle:
        shuffle(all_ts)
    curr_idx = 0
    while curr_idx < len(all_ts):
        # create random batches first
        #batch_paths = np.random.choice(a= all_images, size = batch_size)
        # initialize our batches of images and labels
        #print(all_images[curr_idx])
        ts = []
        labels = []
        curr_batch = all_ts[curr_idx]
        ts, labels = preprocess_test_img(ts_path= curr_batch, class_names = [0,1,2,3,4,5], num_classes = num_classes, epsilon = epsilon, mode=mode)
        ts = np.array(ts)
        labels = np.array(labels)
        curr_idx += batch_size
        yield (ts, labels, curr_batch)

