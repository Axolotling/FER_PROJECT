import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

import csv
import h5py
# PARAMETERS ------------------------------------------------------------------
#import pydot
#import graphviz


im_rows = 48
im_cols = 48

im_shape = (im_rows, im_cols, 1)



N_EPOCHS = 10
BATCH_SIZE = 4444
DATASET_PATH = 'dataset\\dataset.csv'



# DATA PREPARATION ------------------------------------------------------------
data_extracted = False

def extract_data_from_file(path):
    with open(path) as csv_file:
        n_rows = BATCH_SIZE
        row_counter = 0
        readcsv = csv.reader(csv_file,delimiter=',')
        for row in readcsv:
            row_counter += 1
            #Y.append(int(row[0]))
           # tmp_list = [0,0,0,0,0,0,0,0]
           # tmp_list[int(row[0])] = 1
            Y.append(int(row[0]))     
            tmp_list = list(map(int,row[1].split()))
            tmp_list = [(x+1)/256.0 for x in tmp_list]
            tmp_matrix = np.reshape(tmp_list,im_shape)
            #np.reshape()
                
            X.append(tmp_matrix)
            
            if(row_counter >= n_rows):
                return

if (data_extracted == False):
    Y = []
    X = []   
    extract_data_from_file(DATASET_PATH)
    data_extracted = True
    X = np.array(X)
    Y = np.array(Y)
    
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))
# DEFING MODEL ----------------------------------------------------------------

#image = X[151, :].reshape((48, 48))
#plt.imshow(image)
#plt.show()

old_model = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),    
    Flatten(),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
])

old_model2 = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer ='he_normal', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),    
    Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=im_shape),   
    Dropout(0.25),    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')
])    

model = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer ='he_normal', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.25),    
    Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=im_shape),   
    Dropout(0.25),    
    Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=im_shape),   
    Dropout(0.4),    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
])    
    
tensorboard = TensorBoard(
    log_dir=r'logs\{}'.format('cnn_1layer'),
    write_graph=True,
    write_grads=True,
    histogram_freq=1,
    write_images=True,
)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)

model.fit(
    X, Y, batch_size=BATCH_SIZE,
    epochs=N_EPOCHS, verbose=1,
    validation_split=0.1,
    callbacks=[tensorboard]
)

#npredictions = model.predict(X[:40])
class_predictions = model.predict_classes(X[:40])