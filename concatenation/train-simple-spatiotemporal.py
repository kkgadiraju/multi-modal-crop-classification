import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
np.random.seed(100)
tf.set_random_seed(100)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=tf_config))
from tensorflow.python.keras.layers import Input, LSTM, Dense, TimeDistributed, concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from time import time
import configparser, argparse, datetime, json, os
from data_generator import crop_generator 
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.utils import plot_model
import visualize
import socket

curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


networks = {'resnet50': ResNet50,
            'densenet':DenseNet201,
            'vgg16':VGG16,
             'vgg19':VGG19}

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config.ini')
parser.add_argument('-d', '--dropout', help="""dropout rate""", default='0.5')
parser.add_argument('-r', '--runNumber', help="""number of the run for repeated caculations""", default='1')
parser.add_argument('-t', '--task', help="""classification task you are performing - either crop, water or energy""", default='water-lstm')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)


batch_size = int(config[parser_args.task]['BATCH_SIZE'])
tr_path = str(config[parser_args.task]['TRAIN_FOLDER'])
val_path = str(config[parser_args.task]['VAL_FOLDER'])
num_classes = int(config[parser_args.task]['NUM_CLASSES'])
learning_rate = float(config[parser_args.task]['LEARNING_RATE'])
num_epochs = int(config[parser_args.task]['NUM_EPOCHS'])
dropout_rate = float(parser_args.dropout)
epsilon = float(config[parser_args.task]['EPSILON'])
plot_path = str(config['plotting']['PLOT_FOLDER'])
measures_folder = str(config['plotting']['MEASURE_FOLDER'])

target_size = (224, 224)
print("Batch size = {}".format(batch_size))
print(f'Run number = {parser_args.runNumber}')
print(f"Hostname: {socket.gethostname()}, Number of classes = {num_classes}\n Learning rate = {learning_rate}\n dropout rate = {dropout_rate}\n timestamp = {curr_time}") 


train_generator = crop_generator(input_path=tr_path, batch_size=batch_size, mode="train", do_shuffle=True, epsilon=0)

val_generator = crop_generator(input_path=val_path, batch_size=batch_size, mode="val", do_shuffle=True, epsilon=0)


network = networks[str(config[parser_args.task]['NETWORK'])]

# spatial part starts here
cnn_base_model = network(include_top=False, weights = 'imagenet',classes=num_classes)

cnn_output = cnn_base_model.output

img_avgpool = GlobalAveragePooling2D()(cnn_output)

img_dense_1 = Dense(512, activation="relu")(img_avgpool)

img_dropout = Dropout(dropout_rate)(img_dense_1)

img_dense_2 = Dense(256, activation="relu")(img_dropout)
# spatial part ends here

# temporal part starts here
temporal_input_layer = Input(batch_shape = (None, 23, 1), name='time_input_layer')


# add an LSTM layer
lstm_1 = LSTM(100, input_shape=(23,1))(temporal_input_layer)

#lstm_2 = LSTM(64)(lstm_1)
#x = LSTM(32)(x)

# add a dense layer
ts_output = Dense(32, activation="relu")(lstm_1)
# temporal part ends here

# add a concatenation layer to combine output of spatial and temporal
final_merged = concatenate([img_dense_2, ts_output])

# add dense layer
final_dense = Dense(32, activation="relu")(final_merged)

# and a softmax layer -- num_classes
predictions = Dense(num_classes, activation='softmax')(final_dense)

optimizer = Adam(lr=learning_rate)

# this is the model we will train
model = Model(inputs=[cnn_base_model.input, temporal_input_layer], outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

print(model.summary())


## NN ends here

# callbacks for checkpointing
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model_name = str(config[parser_args.task]['MODEL_FOLDER']) + '_' + str(config[parser_args.task]['NETWORK']) + curr_time + '.h5'
model_checkpoint = ModelCheckpoint(model_name, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor="val_loss", verbose=1, patience=8) 
log_file_name = os.path.join(config['plotting']['MEASURE_FOLDER'], 'evals-{}.json'.format(curr_time))
csv_logger = CSVLogger(filename=log_file_name, append = True)

# generate a model by training 
history = model.fit_generator(train_generator,  
			epochs=num_epochs,
			steps_per_epoch=36690//batch_size,
			validation_data= val_generator, 
			validation_steps=12227//batch_size,
			verbose=1, 
			callbacks=[tensorboard, model_checkpoint, early_stopping, csv_logger])


#model history generate plot
visualize.visualize_curves(n_epochs=num_epochs, tr=history.history['loss'], val=history.history['val_loss'], filename='{}/loss_{}.png'.format(plot_path, curr_time), network=str(config[parser_args.task]['NETWORK']), optimizer='adam', learning_rate=learning_rate, epsilon=epsilon, clf_type=parser_args.task, batch_size=batch_size, viz_type="loss", early_stopping=True)

visualize.visualize_curves(n_epochs=num_epochs, tr=history.history['acc'], val=history.history['val_acc'], filename='{}/acc_{}.png'.format(plot_path, curr_time), network=str(config[parser_args.task]['NETWORK']), optimizer='adam', learning_rate=learning_rate, epsilon=epsilon, clf_type=parser_args.task, batch_size=batch_size, viz_type="accuracy", early_stopping=True)

# draw a model
visualize.visualize_model(model=model, filename="{}/model_{}.png".format(plot_path, curr_time))

# save the model history to disk to plot all results in single plot later
# write_measures(history, measures_folder, 'measures_{}'.format(curr_time))

print(f'Run number = {parser_args.runNumber}')
with open('{}-config_{}.ini'.format(config[parser_args.task]['MODEL_FOLDER'], curr_time), 'w') as cfgfile:
     newConfig = configparser.ConfigParser() 
     newConfig[parser_args.task] = config[parser_args.task]
     #print(config[parser_args.task])
     newConfig.write(cfgfile)

print(f"Hostname: {socket.gethostname()}, Number of classes = {num_classes}\n Learning rate = {learning_rate}\n dropout rate = {dropout_rate}\n timestamp = {curr_time}") 
print("Model is saved at: {}".format(model_name)) 
