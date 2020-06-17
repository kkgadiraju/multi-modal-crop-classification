import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
np.random.seed(100)
tf.set_random_seed(100)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=tf_config))
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Dropout, BatchNormalization
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras import regularizers
from sklearn.metrics import classification_report
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from time import time
import configparser, argparse, datetime, visualize, json, os, socket
import pandas as pd
import itertools


start_time_n = datetime.datetime.now()

start_time = start_time_n.strftime("%Y-%m-%d %H:%M:%S")



networks = {'inception': InceptionV3,
            'resnet50': ResNet50,
	    'densenet':DenseNet201,
	    'vgg16':VGG16,
             'vgg19':VGG19}

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config.ini')
parser.add_argument('-t', '--task', help="""task you are performing - refers to the header for each section in the config file""", default='crop-vgg16')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)

batch_size = int(config[parser_args.task]['BATCH_SIZE'])
tr_path = str(config[parser_args.task]['TRAIN_FOLDER'])
val_path = str(config[parser_args.task]['VAL_FOLDER'])
num_classes = int(config[parser_args.task]['NUM_CLASSES'])
learning_rates = [0.0001, 0.0005, 0.00001, 0.00005]#float(config[parser_args.task]['LEARNING_RATE'])
num_epochs = int(config[parser_args.task]['NUM_EPOCHS'])
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]#float(parser_args.dropout)
epsilon = float(config[parser_args.task]['EPSILON'])
plot_path = str(config[parser_args.task]['PLOT_FOLDER'])
l2_reg = float(config[parser_args.task]['L2_REG']) 
train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)


grid_file = './{}_grid.csv'.format(config[parser_args.task])

if os.path.exists(grid_file):
    grid_df = pd.read_csv(grid_file)

else:
    param_grid = list(itertools.product(learning_rates, dropout_rates))
    grid_df = pd.DataFrame(param_grid, columns = ['lr', 'd'])
    grid_df['timestamp'] = 'None'

print(grid_df)


if config[parser_args.task]['NETWORK'] == "inception":
    target_size = (299, 299)
else:
    target_size = (224, 224)

grid = ((lr, d) for lr in learning_rates for d in dropout_rates)

for idx, row in grid_df.iterrows():
    if row['timestamp'] == 'None':
        learning_rate = row['lr']
        dropout_rate = row['d']
        curr_time_n = datetime.datetime.now()
        curr_time = curr_time_n.strftime("%Y-%m-%d %H:%M:%S")
        time_passed_since_start = curr_time_n - start_time_n
        time_passed_since_start_in_min = time_passed_since_start/datetime.timedelta(minutes=1)
        time_passed_since_start_in_hours = time_passed_since_start_in_min/60

        if time_passed_since_start_in_hours >= 240.:
            print('Not enough time for lr = {}, dropout = {}'.format(learning_rate, dropout_rate))
            grid_df.to_csv(grid_file, index=False)
            sys.exit(0)
        else:
            grid_df.loc[idx, 'timestamp'] = curr_time
            #print(f"Hostname: {socket.gethostname()}, Number of classes = {num_classes}\n Learning rate = {learning_rate}\n dropout rate = {dropout_rate}\n timestamp = {curr_time}")
            train_generator = train_datagen.flow_from_directory(tr_path, target_size=target_size, batch_size=batch_size,class_mode='categorical', shuffle=True)

            val_generator = val_datagen.flow_from_directory(val_path, target_size=target_size, batch_size=batch_size,class_mode='categorical', shuffle=True)

            # NN starts here

            network = networks[str(config[parser_args.task]['NETWORK'])]

            if parser_args.task == "crop-henry-{}".format(batch_size):
                base_model = network(include_top=False, weights = '/home/kgadira/Food-Water-Energy/7_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',classes=num_classes)

            else:
                base_model = network(include_top=False, weights = 'imagenet',classes=num_classes)

            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            #  two fully-connected layers
            if l2_reg > 0.:
                x = Dense(512, activation='relu',  kernel_regularizer=regularizers.l2(l2_reg))(x)
                # add a dropout layer
                x = Dropout(dropout_rate)(x)
                x = Dense(256, activation='relu',  kernel_regularizer=regularizers.l2(l2_reg))(x)
            else:
                x = Dense(512, activation='relu')(x)
                # add a dropout layer
                x = Dropout(dropout_rate)(x)
                x = Dense(256, activation='relu')(x)

            # and a softmax layer -- 6 classes
            predictions = Dense(num_classes, activation='softmax')(x)

            optimizer = Adam(lr=learning_rate, epsilon=epsilon)

            # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)


            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

            ## NN ends here

            # callbacks for checkpointing
            #tensorboard = TensorBoard(log_dir="logs/{}".format(time()), update_freq='batch')
            model_name = str(config[parser_args.task]['MODEL_FOLDER']) + '_' + str(config[parser_args.task]['NETWORK']) + curr_time + '.h5'
            model_checkpoint = ModelCheckpoint(model_name, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
            early_stopping = EarlyStopping(monitor="val_loss", verbose=1, patience=8) 
            log_file_name = os.path.join(config[parser_args.task]['MEASURE_FOLDER'], 'evals-{}.json'.format(curr_time))
            csv_logger = CSVLogger(filename=log_file_name, append = True)

            # generate a model by training 
            history = model.fit_generator(train_generator,  
			epochs=num_epochs,
			steps_per_epoch=36690//batch_size,
			validation_data= val_generator, 
			validation_steps=12227//batch_size,
			verbose=1, 
			callbacks=[early_stopping, model_checkpoint, csv_logger])

            # since we are checkpointing best model, don't need to save
            # model.save(model_name)

            # model history generate plot
            #visualize.visualize_curves(n_epochs=num_epochs, tr=history.history['loss'], val=history.history['val_loss'], filename='{}/loss_{}.png'.format(plot_path, curr_time), network=str(config[parser_args.task]['NETWORK']), optimizer='adam', learning_rate=learning_rate, epsilon=epsilon, clf_type=parser_args.task, batch_size=batch_size, viz_type="loss", early_stopping=True)

            #visualize.visualize_curves(n_epochs=num_epochs, tr=history.history['acc'], val=history.history['val_acc'], filename='{}/acc_{}.png'.format(plot_path, curr_time), network=str(config[parser_args.task]['NETWORK']), optimizer='adam', learning_rate=learning_rate, epsilon=epsilon, clf_type=parser_args.task, batch_size=batch_size, viz_type="accuracy", early_stopping=True)

            # draw a model
            #visualize.visualize_model(model=model, filename="{}/model_{}.png".format(plot_path, curr_time))

            # save the model history to disk to plot all results in single plot later
            # write_measures(history)

            config[parser_args.task]['DROPOUT'] = str(dropout_rate)
            with open('{}-config_{}.ini'.format(config[parser_args.task]['MODEL_FOLDER'], curr_time), 'w') as cfgfile:
                newConfig = configparser.ConfigParser() 
                newConfig[parser_args.task] = config[parser_args.task]
                newConfig.write(cfgfile)

            #print(f"Hostname: {socket.gethostname()},\n Number of classes = {num_classes}\n Learning rate = {learning_rate}\n dropout rate = {dropout_rate}\n timestamp = {curr_time}") 
            print("Model is saved at: {}".format(model_name))  

            # write grid to disk during current iteration to ensure that temporary results are stored
            grid_df.to_csv(grid_file, index=False)

