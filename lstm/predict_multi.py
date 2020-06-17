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


timestamps = ['crop_temporal'+x+'.h5' for x in ['2019-11-15 19:25:15']]

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config-gs.ini')
parser.add_argument('-t', '--task', help="""task you are performing - refers to the header for each section in the config file""", default='temporal')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)
			
te_path = str(config[parser_args.task]['TEST_FOLDER'])

test_accuracies = []
kappa_scores = []

for timestamp in timestamps:
    model_name = os.path.join('/home/kgadira/multi-modal-crop-classification/8_models/', timestamp)
    test_datagen = test_crop_generator(input_path=te_path, batch_size=1, mode="test", num_classes =6, epsilon = 0, resize_params = (224, 224), do_shuffle=True)

    print('Loading model {}'.format(model_name))
    
    predicted_probabilities_csv_name = './{}-probs.csv'.format(config[parser_args.task])

    model = load_model(model_name)

    print(model.summary())
    all_predictions = []
    all_gt = []
    data_paths = []
    results = []
    for te_data, label, curr_path in test_datagen:
        result = model.predict(te_data, verbose=0)
        results.append(result)
        prediction = np.argmax(result, axis = 1)
        all_predictions.append(prediction)
        all_gt.append(label)
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


    test_accuracies.append(acc)
    
    print(creport_df)
    print('Accuracy for {} is {}, kappa score is {}'.format(timestamp, acc, kappa_score))


    predict_df = pd.DataFrame(data=results, columns=['TP0', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5'])


    predict_df.to_csv(predicted_probabilities_csv_name)

                
    visualize.plot_confusion_matrix(cm, classes=['Corn', 'Cotton', 'Soy', 'Spring Wheat', 'Winter Wheat', 'Barley'], title='t-lstm Confusion Matrix')
     
