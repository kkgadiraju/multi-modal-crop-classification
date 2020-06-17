import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=tf_config))

from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
import configparser, argparse, datetime, json, os, visualize
import pandas as pd
from data_generator import test_crop_generator


timestamps = ['crop-bilstm_temporal'+x+'.h5' for x in ['2020-02-11 02:04:33']]

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config-gs.ini')
parser.add_argument('-t', '--task', help="""task you are performing - refers to the header for each section in the config file""", default='temporal-bilstm')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)
			
tr_path = str(config[parser_args.task]['TRAIN_FOLDER'])
val_path = str(config[parser_args.task]['VAL_FOLDER'])
te_path = str(config[parser_args.task]['TEST_FOLDER'])

test_accuracies = []
for timestamp in timestamps:
    model_name = os.path.join('/home/kgadira/multi-modal-crop-classification/8_models/', timestamp)
    test_datagen = test_crop_generator(input_path=te_path, batch_size=1, mode="test", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True)
    print('Loading model {}'.format(model_name))

    model = load_model(model_name)
  
    print(model.summary()) 
    for data_type, curr_gen in zip(['test'], [test_datagen]):
    
        all_predictions = []
        all_gt = []
        data_paths = []
        results = []
        for te_data, label, curr_path in curr_gen:
            result = model.predict(te_data, verbose=0)
            results.append(result)
            prediction = np.argmax(result, axis = 1)
            all_predictions.append(prediction)
            all_gt.append(label[0])
            data_paths.append(curr_path)
        results = np.array(results)
        results = np.squeeze(results, axis=1)

        cm = confusion_matrix(all_gt, all_predictions)

        print(cm)



        classes_lst = ['Corn', 'Cotton', 'Soy', 'Spring Wheat', 'Winter Wheat', 'Barley']

        creport = classification_report(y_true = all_gt, y_pred=all_predictions, target_names = classes_lst, digits = 4, output_dict = True)

        creport_df = pd.DataFrame(creport).transpose()

        acc = accuracy_score(all_gt, all_predictions)

        kappa_score = cohen_kappa_score(all_gt, all_predictions) 
        print(creport_df)
        print('Accuracy for {} is {}, Kappa Score is {}'.format(timestamp, acc, kappa_score))


        #predict_df = pd.DataFrame(data=results, columns=['TP0', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5'])

        #file_names = [x.split('/')[-1] for x in data_paths]

        #predict_df['fname'] = file_names

        #predict_df['Class'] = all_gt

        #predict_df.to_csv(predicted_probabilities_csv_name, index=False)

        visualize.plot_confusion_matrix(cm, classes=classes_lst, title='t-bilstm Confusion Matrix')
                

#print(f'All test accuracies = {test_accuracies}')

#creport_df.to_csv(parser_args.network+'creport.csv', index = True)
 
