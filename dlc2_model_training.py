#Import image transformation packages
import skimage
from skimage import filters, io, exposure, color, segmentation, feature, morphology
from skimage.feature import canny
from scipy import ndimage as ndi
from scipy import misc
import skimage.transform as skt

#Import required packages
import numpy as np
import pandas as pd
import os
import glob

#Import Visualisation packages
import matplotlib.pyplot as plt
import graphviz

#Import sklearn modules
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

#Import Keras
import keras
from keras import metrics
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy,sparse_categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, Concatenate, Dense, Dropout, Flatten, Activation, Merge
from keras.models import Model,load_model
from keras.utils import plot_model
from keras import callbacks
import keras.backend as K

#Change working directory
os.chdir('/home/sambeet/data/hackerearth/deep learning challenge 2/')

#Read train, test and validation packages
#split_string = lambda x : x.split('_')[1]
train_data = pd.read_csv('csv/train_data.csv')
test_data = pd.read_csv('csv/test_data.csv')
val_data = pd.read_csv('csv/val_data.csv')
#train_data.detected = train_data.detected.apply(split_string)
#test_data.detected = test_data.detected.apply(split_string)
#all_data = pd.read_csv('csv/train.csv')
encoder = LabelEncoder()
encoder.fit(train_data.detected.values)
encoded_2 = encoder.transform(train_data.detected.values)

class_weights = class_weight.compute_class_weight('balanced', np.unique(encoded_2), encoded_2)
labels_dict = dict()
for key in np.unique(encoded_2):
    labels_dict[key] = class_weights[key]

def data_generator(batch_size = 8, dataset = 'train'):
    if dataset == 'train':
        df = train_data.copy()
    elif dataset == 'test':
        df = test_data.copy()
    else:
        df = val_data.copy()
    
    df = df.sample(frac=1).reset_index(drop=True)
    image_list = list(df.image_name.values)
    numeric_variables = df[['age','gender_M','view_position']].values
    # encode class values as integers
    encoded_Y = encoder.transform(df.detected.values)
    # convert integers to dummy variables (i.e. one hot encoded)
    labels = keras.utils.to_categorical(encoded_Y)
    while 1:
        for batch_num in range(len(image_list)//batch_size):
            start_index = batch_num*batch_size
            end_index = (batch_num + 1)*batch_size
            batch_images = image_list[start_index:end_index]
            numeric_data_1 = numeric_variables[start_index:end_index]
            images = np.empty((batch_size, 512, 512, 3), dtype = np.float32)
            numeric_data_2 = np.empty((batch_size,8), dtype = np.float32)
            detected = labels[start_index:end_index]
            for i,image_name in zip(range(batch_size),batch_images):
                ima_copy = misc.imread('train_resized/512/' + image_name,flatten = True)/255.
                images[i,...,0] = ima_copy
                images[i,...,1] = ima_copy
                images[i,...,2] = ima_copy
                numeric_data_2[i,...] = np.histogram(images[i,...], bins = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9,1])[0][0:8]/float(512*512)
                i = i + 1
            numeric_data = np.hstack((numeric_data_1,numeric_data_2))
            yield [images,numeric_data], detected

train_data_gen = data_generator(16,'train')
test_data_gen = data_generator(4,'test')

image_input = Input(shape=(512, 512, 3))
x = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=image_input, pooling=None, classes=14)
for layer in x.layers:
    layer.trainable = False

#Adding custom Layers 
y = x.output
y = Flatten()(y)
input2 = Input(shape=(11,))
merged_input = keras.layers.concatenate([y, input2])
y = Dense(1024, activation="relu")(merged_input)
y = Dropout(0.5)(y)
y = Dense(1024, activation="relu")(y)
predictions = Dense(14, activation="softmax")(y)

# creating the final model 
model = Model(inputs= [x.input,input2], outputs = predictions)

# compile the model
model.compile(loss = "categorical_crossentropy", optimizer = Adam(1e-5), metrics=["accuracy"])
model.summary()

#plot_model(model, to_file='dl2_model_1.png')

model_checkpoint = callbacks.ModelCheckpoint(filepath = 'logs/resnet50_1e-5.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
tensorboard = callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=2, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
csv_logger = callbacks.CSVLogger('logs/training.log')

model.load_weights('resnet_512_tl_40eps_1e-5.hd5')#,custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})

history = model.fit_generator(train_data_gen, epochs=48, steps_per_epoch= 464*2, verbose = 1,initial_epoch=40,
                                callbacks=[model_checkpoint,tensorboard,csv_logger],class_weight = labels_dict,
                                validation_data = test_data_gen, validation_steps = 464)

model.save('resnet_512_tl_48eps_1e-5.hd5')
