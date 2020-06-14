
# Write custom data generator for training and testing our CNN network for crop classification using label smoothing
#
######

import numpy as np
#np.random.seed(100)
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
    #print(datapoint, class_names)
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


def preprocess_batches(image_batch, class_names, num_classes, epsilon, resize_width_and_height, mode):
    return_x_img = []
    return_x_ts = []
    return_y = []
    for image_path in image_batch:
        curr_class = int(os.path.basename(os.path.dirname(image_path)))
        curr_file_name = os.path.basename(image_path).split('.')[0]
        curr_ts_path = os.path.join(str(Path(image_path).parent.parent.parent)+'-ts', mode, str(curr_class), '{}.csv'.format(curr_file_name))
        curr_ts = pd.read_csv(curr_ts_path)
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
        curr_img = Image.open(image_path).resize(resize_width_and_height, Image.NEAREST)
        curr_img = np.array(curr_img)
        curr_img = curr_img * (1./255.) 
        return_x_img.append(curr_img)
        return_x_ts.append(curr_ts_data)
    return_x_ts = np.array(return_x_ts, dtype=np.float32)
    assert len(return_x_img) == len(return_y), "TrainingX images and Y lengths mismatch!"
    assert len(return_x_ts) == len(return_y), "TrainingX time series and Y lengths mismatch!"
    return [return_x_img, return_x_ts, return_y]        

def preprocess_test_img(image_path, class_names, num_classes, epsilon, resize_width_and_height, mode):
    return_x_img = []
    return_x_ts = []
    return_y = []
    curr_class = int(os.path.basename(os.path.dirname(image_path)))
    curr_file_name = os.path.basename(image_path).split('.')[0]
    curr_ts_path = os.path.join(str(Path(image_path).parent.parent.parent)+'-ts', mode, str(curr_class), '{}.csv'.format(curr_file_name))
    curr_ts = pd.read_csv(curr_ts_path)
    if curr_ts.empty:
        print("{} seems to be empty: pls check ".format(ts_path))
        sys.exit(0)
    curr_ts_data = curr_ts['NDVI'].values
    curr_ts_data = replace_nans(curr_ts_data)
    curr_ts_data = curr_ts_data[:, np.newaxis]
    curr_ts_data = StandardScaler().fit_transform(curr_ts_data)
    return_y.append(curr_class)
    curr_img = Image.open(image_path).resize(resize_width_and_height, Image.NEAREST)
    curr_img = np.array(curr_img) 
    curr_img = curr_img * (1./255.)
    return_x_img.append(curr_img)
    return_x_ts.append(curr_ts_data)
    return_x_ts = np.array(return_x_ts, dtype=np.float32)
    assert len(return_x_img) == len(return_y), "TrainingX images and Y lengths mismatch!"
    assert len(return_x_ts) == len(return_y), "TrainingX time series and Y lengths mismatch!"
    return [return_x_img, return_x_ts, return_y]        


"""
def preprocess_ts_batches(ts_batch, class_names, num_classes, epsilon):
    return_x = []
    return_y = []
    for ts_path in ts_batch:
        curr_class = int(os.path.basename(os.path.dirname(ts_path)))
        curr_class_encoded = one_hot_encode(class_names, curr_class)  
        curr_class_label_smoothed = label_smooth(num_classes = num_classes, epsilon = epsilon, encoded_data_point = curr_class_encoded)
        return_y.append(curr_class_label_smoothed)
        #print(os.path.isfile(ts_path))
        curr_ts = pd.read_csv(ts_path)
        if curr_ts.empty:
            print("{} seems to be empty: pls check ".format(ts_path))
        curr_data = curr_ts['NDVI'].values
        curr_data = replace_nans(curr_data)
        curr_data = curr_data[:, np.newaxis]
        curr_data = StandardScaler().fit_transform(curr_data)
        return_x.append(curr_data)
    return_x = np.array(return_x, dtype=np.float32)
    #print(return_x)
    #return_x = return_x[:, :,  np.newaxis]
    return_y = np.array(return_y, dtype = np.float32)
    assert return_x.shape[0] == return_y.shape[0], "TrainingX and Y lengths mismatch!"
    return [return_x, return_y]        
"""
    
def crop_generator(input_path, batch_size=32, mode="train", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True):	
    """
    Simple data generator that reads all images based on mode, picks up corresponding time series, returns entire list
    """
    data_path = os.path.join(input_path, mode)
    all_images = glob.glob(os.path.join(data_path, "**/*.jpg"))
    #all_ts = glob.glob(os.path.join(data_path, "**/*.csv"))
    print("Found {} files for {}".format(len(all_images), mode))
    if do_shuffle:
        shuffle(all_images)
    curr_idx = 0
    while True:
        # create random batches first
        #batch_paths = np.random.choice(a= all_images, size = batch_size)
        # initialize our batches of images and labels
        imgs = []
        ts = []
        labels = []       
        if curr_idx > len(all_images): # reset if you've parsed all data
            curr_idx = 0
        curr_batch = all_images[curr_idx: (curr_idx + batch_size)]
        _, ts, labels = preprocess_batches(image_batch= curr_batch, class_names = [0,1,2,3,4,5], num_classes = num_classes, epsilon = epsilon, resize_width_and_height=resize_params, mode=mode) 
        ts = np.array(ts)
        labels = np.array(labels)
        curr_idx += batch_size 
        yield (ts, labels)

def test_crop_generator(input_path, batch_size=1, mode="test", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True):	
    """
    Simple data generator that reads all images based on mode, picks up corresponding time series, returns entire list
    """
    data_path = os.path.join(input_path, mode)
    all_images = glob.glob(os.path.join(data_path, "**/*.jpg"))
    print("Found {} files for {}".format(len(all_images), mode))
    if do_shuffle:
        shuffle(all_images)
    curr_idx = 0
    while curr_idx < len(all_images):
        # create random batches first
        #batch_paths = np.random.choice(a= all_images, size = batch_size)
        # initialize our batches of images and labels
        #print(all_images[curr_idx])
        imgs = []
        ts = []
        labels = []       
        curr_batch = all_images[curr_idx]
        _, ts, labels = preprocess_test_img(image_path= curr_batch, class_names = [0,1,2,3,4,5], num_classes = num_classes, epsilon = epsilon, resize_width_and_height=resize_params, mode=mode) 
        #imgs = np.array(imgs)
        ts = np.array(ts)
        labels = np.array(labels)
        curr_idx += batch_size
        yield (ts, labels, curr_batch)

# res = crop_generator('/mnt/data/kgadira/food-water-energy/filtered-extracts-subset')
# for a in res:
#    print(a)


