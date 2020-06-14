#!/bin/sh
source activate cnn-rnn
#python train-simple-spatiotemporal.py -c './config.ini' -t 'spatiotemporal' -d $1 -r $2
python train-gridsearch.py -c './config-gs.ini' -t 'temporal' 
