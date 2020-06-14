#!/bin/sh
source activate cnn-rnn
python train.py -c './config.ini' -t 'spatiotemporal' 
