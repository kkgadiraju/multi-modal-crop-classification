#!/bin/sh
source activate cnn-rnn
python train-gridsearch.py -c './config-gs.ini' -t 'temporal' 
