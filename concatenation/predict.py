import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import configparser, argparse, datetime, json, os, visualize
import pandas as pd
from data_generator import test_crop_generator

#networks = {'inception': 'crop_inception2019-04-07 21:11:45.h5',
networks = { 'resnet50': 'crop_resnet502019-08-25 22:46:31.h5',
            #'densenet':'crop_densenet2019-08-05 18:16:28.h5',
            'vgg16':'crop_vgg162019-08-23 14:10:43.h5'}

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config.ini')
parser.add_argument('-t', '--task', help="""classification task you are performing - either crop or energy""", default='crop-gad')
parser.add_argument('-n', '--network', help="""name of the network you are predicting for""", default='vgg16')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)
			
te_path = str(config[parser_args.task]['TEST_FOLDER'])
model_name = os.path.join('/mnt/data/kgadira/food-water-energy/model/', networks[str(parser_args.network)])

if str(parser_args.network) == "inception":
    input_size = 299

else:
    input_size = 224

test_datagen = test_crop_generator(input_path=te_path, batch_size=1, mode="test", num_classes =4, epsilon = 0, resize_params = (224, 224), do_shuffle=True)
#test_generator = test_datagen.flow_from_directory(te_path, target_size = (input_size, input_size), class_mode='categorical', shuffle = False, batch_size=1)

print('Loading model {}'.format(model_name))

model = load_model(model_name)

all_predictions = []
all_gt = []
data_paths = []
for te_data, label, curr_path in test_datagen:
    result = model.predict(te_data, verbose=0)
    #print(result.shape)
    prediction = np.argmax(result, axis = 1)
    #print(prediction, label)
    all_predictions.append(prediction)
    all_gt.append(label)
    data_paths.append(curr_path)
#print(all_predictions)

#print(all_gt)

"""
actual_labels = test_generator.classes

"""
cm = confusion_matrix(all_gt, all_predictions)

print(cm)



classes_lst = ['Corn', 'Cotton', 'Soy', 'Wheat']

creport = classification_report(y_true = all_gt, y_pred=all_predictions, target_names = classes_lst, digits = 4, output_dict = True)

creport_df = pd.DataFrame(creport).transpose()

acc = accuracy_score(all_gt, all_predictions)

# f1 = f1_score(actual_labels, predictions, labels = classes_lst, average='samples')

print(creport)
print(acc)

visualize.plot_confusion_matrix(cm, classes=['Corn', 'Cotton', 'Soy', 'Wheat'], title=parser_args.network+' Confusion Matrix')

creport_df.to_csv(parser_args.network+'creport.csv', index = True)
 
for i in range(len(all_gt)):
    if int(all_predictions[i]) != int(all_gt[i]):
        print('Path = {}, Actual = {}. Predicted = {}'.format(data_paths[i], all_gt[i], all_predictions[i]))
        
