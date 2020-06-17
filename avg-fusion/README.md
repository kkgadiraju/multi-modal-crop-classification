## Summary
Averages the spatial and temporal streams 


## How to run the code:

1. Download the data, collect its path and update it in ```config-gs.ini``` folder.

2. To run training/gridsearch cd to each folder and do:
 
 ```python3 train_gridsearch.py --config '.config file name' --task 'task name' --network 'network name'```


    * ```--config``` refers to the config file
    * ```--task``` here refers to the header in the config file. 
    * ```--network``` refers to the name of backbone network for spatial stream (vgg16, resnet50, densenet etc).
    * You can specify the parameters in the ```train_gridsearch.py``` file. For each combination of parameters specified in ```train_gridsearch.py```, it generates a model as per specifications and saves it to disk.

3. Note that the models are given names based on timestamp when the training was started. This gives us easy way to check which hyperparameters in grid gave best result and choose that for prediction. For each run, the training losses etc are all saved using this time stamp, for easy correlation. 

4. To change configuration parameters like input/output paths etc. update them in config file

5. To evalulate a particular model, update the training model file names you created above in the first few lines of ```predict_multi.py``` and run the code as shown below: 
> ``` python3 predict_multi.py --config 'config file name' --task 'task name' --network 'network_name'```
