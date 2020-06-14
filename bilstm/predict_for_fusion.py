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
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config.ini')
parser.add_argument('-t', '--task', help="""classification task you are performing - either crop or energy""", default='crop-gad')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)
			
tr_path = str(config[parser_args.task]['TRAIN_FOLDER'])
val_path = str(config[parser_args.task]['VAL_FOLDER'])
te_path = str(config[parser_args.task]['TEST_FOLDER'])

test_accuracies = []
for timestamp in timestamps:
    model_name = os.path.join('/mnt/data3/crop-classification/8_models/', timestamp)
    print(model_name)
    #train_datagen = test_crop_generator(input_path=tr_path, batch_size=1, mode="train", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True)
    #val_datagen = test_crop_generator(input_path=val_path, batch_size=1, mode="val", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True)
    test_datagen = test_crop_generator(input_path=te_path, batch_size=1, mode="test", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True)
    print('Loading model {}'.format(model_name))

    model = load_model(model_name)
  
    print(model.summary()) 
    #for data_type, curr_gen in zip(['train', 'val', 'test'], [train_datagen, val_datagen, test_datagen]):
    for data_type, curr_gen in zip(['test'], [test_datagen]):
    
        predicted_probabilities_csv_name = './pt-bilstm-{}-{}-probs.csv'.format(data_type, config[parser_args.task])


        all_predictions = []
        all_gt = []
        data_paths = []
        results = []
        for te_data, label, curr_path in curr_gen:
            result = model.predict(te_data, verbose=0)
            results.append(result)
            #print(result.shape)
            prediction = np.argmax(result, axis = 1)
            #print(prediction, label)
            all_predictions.append(prediction)
            all_gt.append(label[0])
            data_paths.append(curr_path)
        #print(all_predictions)
        results = np.array(results)
        print(results.shape)
        results = np.squeeze(results, axis=1)

        cm = confusion_matrix(all_gt, all_predictions)

        print(cm)



        classes_lst = ['Corn', 'Cotton', 'Soy', 'Spring Wheat', 'Winter Wheat', 'Barley']

        creport = classification_report(y_true = all_gt, y_pred=all_predictions, target_names = classes_lst, digits = 4, output_dict = True)

        creport_df = pd.DataFrame(creport).transpose()

        acc = accuracy_score(all_gt, all_predictions)

        # f1 = f1_score(actual_labels, predictions, labels = classes_lst, average='samples')

        kappa_score = cohen_kappa_score(all_gt, all_predictions) 
        print(creport_df)
        print('Accuracy for {} is {}, Kappa Score is {}'.format(timestamp, acc, kappa_score))

        print(cm)

        predict_df = pd.DataFrame(data=results, columns=['TP0', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5'])

        file_names = [x.split('/')[-1] for x in data_paths]

        predict_df['fname'] = file_names

        predict_df['Class'] = all_gt

        predict_df.to_csv(predicted_probabilities_csv_name, index=False)

        #for i in range(len(all_gt)):
        #    if int(all_predictions[i]) != int(all_gt[i]):
        #        print('Path = {}, Actual = {}. Predicted = {}'.format(data_paths[i], all_gt[i], all_predictions[i]))
        visualize.plot_confusion_matrix(cm, classes=classes_lst, title='t-bilstm Confusion Matrix')
                

#print(f'All test accuracies = {test_accuracies}')

#creport_df.to_csv(parser_args.network+'creport.csv', index = True)
 
