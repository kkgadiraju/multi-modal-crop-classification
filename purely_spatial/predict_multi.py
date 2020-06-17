'''
@author: Krishna Karthik Gadiraju
Replace the filenames of the best models you found in networks

'''

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=tf_config))
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
import configparser, argparse, datetime, visualize, json, os
import pandas as pd



networks = { 'resnet50': ['crop_resnet50' + x + '.h5' for x in ['2019-12-26 06:02:10']],
            'densenet':['crop_densenet' + x + '.h5' for x in ['2020-01-01 01:57:07']],
            'vgg16':['crop_vgg16'+x+'.h5' for x in ['2019-12-14 17:59:47']]} 

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config.ini')
parser.add_argument('-t', '--task', help="""task you are performing - refers to the header for each section in the config file""", default='crop-vgg16')
parser.add_argument('-n', '--network', help="""name of the network you are predicting for""", default='vgg16')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)
			
te_path = str(config[parser_args.task]['TEST_FOLDER'])

input_size = 224

all_test_accuracies = []
all_kappa_scores = []

for timestamp in networks[parser_args.network]:
    model_name = os.path.join('/home/kgadira/multi-modal-crop-classification/8_models/', timestamp)
    predicted_probabilities_csv_name = './{}-probs.csv'.format(config[parser_args.task])
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(te_path, target_size = (input_size, input_size), class_mode='categorical', shuffle = False, batch_size=1)
    model = load_model(model_name)

    print(model.summary())
    data_paths = test_generator.filenames

    results = model.predict_generator(test_generator, 
				steps=int(test_generator.samples/1.),
				verbose=1)

    predictions = np.argmax(results, axis = 1)

    actual_labels = test_generator.classes

    cm = confusion_matrix(actual_labels, predictions)

    print(cm)

    classes_lst = ['Corn', 'Cotton', 'Soy', 'Spring Wheat', 'Winter Wheat', 'Barley']

    creport = classification_report(y_true = actual_labels, y_pred=predictions, target_names = classes_lst, digits = 4, output_dict = True)

    creport_df = pd.DataFrame(creport).transpose()

    acc = accuracy_score(actual_labels, predictions)

    all_test_accuracies.append(acc)
   
    kappa_score = cohen_kappa_score(actual_labels, predictions)

    all_kappa_scores.append(kappa_score)
    print(creport_df)
    print('Accuracy for {} is {}, kappa score is {}'.format(timestamp, acc, kappa_score))

    
    predict_df = pd.DataFrame(data=results, columns=['SP0', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5'])

    file_names = [x.split('/')[-1] for x in data_paths]
  
    predict_df['fname'] = file_names
 
    predict_df['Class'] = actual_labels
   
    # uncomment to save the predicted probabilites to use in svm
    #predict_df.to_csv(predicted_probabilities_csv_name)

    visualize.plot_confusion_matrix(cm, classes=classes_lst, title=parser_args.network+' Confusion Matrix')

