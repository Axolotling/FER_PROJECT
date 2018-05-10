import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import h5py
#import pydot
#import graphviz

# PARAMETERS ------------------------------------------------------------------


N_EPOCHS = 4
BATCH_SIZE = 1000
DATASET_PATH = 'dataset\\fer2013.csv'



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
            tmp_list = [0,0,0,0,0,0,0,0]
            tmp_list[int(row[0])] = 1
            Y.append(tmp_list)     
            tmp_list = list(map(int,row[1].split()))
            tmp_list = [(x+1)/256.0 for x in tmp_list]
            tmp_matrix = np.reshape(tmp_list,(48,48,1))
            #np.reshape()
            
            X.append(tmp_matrix)
            
            if(row_counter >= n_rows):
                return

if (data_extracted == False):
    Y = []
    X = []   
    extract_data_from_file(DATASET_PATH)
    data_extracted = True
        


# DEFING MODEL ----------------------------------------------------------------


model = tf.keras.Sequential([
  #tf.keras.layers.Dense(1000, "sigmoid", input_shape=(48,48,)),  # intput shape required
  tf.keras.layers.Conv2D(1, [3,3], strides=(1, 1), padding='valid', input_shape=(None,48,48), data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
  tf.keras.layers.Conv2D(1, [3,3], strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
  tf.keras.layers.Conv2D(1, [3,3], strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
  tf.keras.layers.Conv2D(1, [3,3], strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
  #tf.keras.layers.Dense(400, "relu"),
  #tf.keras.layers.Dense(100, "relu"),
  #tf.keras.layers.Dense(25, "relu"),
  tf.keras.layers.Dense(8, "sigmoid")
])

    
    
#tf.keras.utils.plot_model(model)

# Training --------------------------------------------------------------------

def my_own_training(model, X, Y):
    n_batches = 100
    batch_size = 100
    loss_chart = []
    score_chart = []
    plt.axis([0, n_batches, 0, 1])
    #plt.show()
    for i in range(0,n_batches):
        a,b = model.train_on_batch(X[i*batch_size:(i+1)*batch_size],Y[i*batch_size:(i+1)*batch_size])       
        loss_chart.append(a)
        score = model.evaluate(X[n_batches*batch_size:n_batches*batch_size + 50], Y[n_batches*batch_size:n_batches*batch_size + 50], verbose=0)
        score_chart.append(score[0])    
        
    print('loss')    
    plt.plot(loss_chart)
    plt.show('loss per batch')
    plt.plot(score_chart)
    print('loss from model.evaluate()')
    plt.show('score per batch')
   
    
    
# Compile model
#sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#my_own_training(model,X,Y)

model.fit(np.array(X[:1]), Y, epochs=N_EPOCHS, BATCH_SIZE=BATCH_SIZE,  verbose=1, validation_split=0.2,shuffle = True)
model_save_path = "model/saved_model_conv.h5py"
tf.keras.models.save_model(
    model,
    model_save_path,
    overwrite=True,
    include_optimizer=True
)
npredictions = model.predict(X[:40])
#proba_predictions = tf.predict_proba(X[:40], batch_size=32, verbose=1)
class_predictions = model.predict_classes(X[:40])# -*- coding: utf-8 -*-

