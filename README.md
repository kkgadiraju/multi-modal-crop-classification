# Multi Modal Deep Learning Based Crop Classification Using Multispectral and Multitemporal Satellite Imagery

## Summary
This repository contains the code for the paper Multi Modal Deep Learning Based Crop Classification Using Multispectral and Multitemporal Satellite Imagery that will be published as a poster paper in KDD Applied Data Science Track 2020.


## Data
Data can be downloaded from [here](https://bit.ly/2ORb16U)


## Software Configuration
    
* python==3.7.4
* tensorflow-gpu==1.13.1
* keras==2.2.4
* sklearn==0.21.2
* numpy==1.16.4
* matplotlib==3.1.1
* pandas==0.25.1
* configparser    
    


## How to run the code:


The following gives the folder descriptions. Each folder is a separate set of experiment:

* ```purely_spatial```: purely spatial 2D crop image classification using well-known neural networks.

* ```lstm```, ```bi-lstm``` and ```1dcnn```: purely temporal crop time series classification that only uses the temporal part of our data.

* ```concatenate```: concantenates purely spatial and purely temporal streams.

* ```avg-fusion```: uses average to combine the spatial and temporal streams.

* ```svm-fusion```: uses SVM classification to predict on a concatenation of spatial and temporal stream predicted probabilites.   

To run the experiments, each folder has a readme file with instructions. 

