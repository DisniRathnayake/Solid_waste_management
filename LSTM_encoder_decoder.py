# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import os
import sys
from math import sqrt
from numpy import split
from numpy import array
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
#from keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from sklearn.preprocessing import StandardScaler

if len(sys.argv) < 5:
    print('Error: argv length mismatch :', len(sys.argv))
    print('sys.argv[1] : VERBOSE')
    print('sys.argv[2] : BATCH_SIZE')
    print('sys.argv[3] : EPOCHS')
    print('sys.argv[4] : #LSTM_hidden_size')
    sys.exit(0)
    
VERBOSE = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])  # to train per epoch
EPOCHS = int(sys.argv[3])
LSTM_units = int(sys.argv[4])

scaler = StandardScaler()
# split a univariate dataset into train/test sets
def split_dataset(data):
    #data = scaler.fit_transform(data)
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    # restructure into windows of weekly data
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        if mse<0:
            mse = (-1)*mse
        else:
            mse = mse
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [Overall RMSE: %.3f] RMSE per day: %s' % (name, score, s_scores))
    
# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    #train = scaler.inverse_transform(train)
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)

def to_valdate(test, n_input, n_out=7):
    # flatten data
    #test = scaler.inverse_transform(test)
    data = test.reshape((test.shape[0]*test.shape[1], test.shape[2]))
   
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)

# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    test_x, test_y = to_valdate(test, n_input)
    # define parameters
    verbose, epochs, batch_size = VERBOSE,EPOCHS,BATCH_SIZE
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    # define model
    model = tf.keras.Sequential()
    model.add(LSTM(LSTM_units, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(LSTM(LSTM_units, activation='relu', return_sequences=False))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(LSTM_units, activation='relu', return_sequences=True))
    model.add(LSTM(LSTM_units, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    model.summary()
    # fit network
    tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="tb_logs",histogram_freq=1)
    model.fit(train_x, train_y,validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=[tb_callbacks],shuffle=True)
    return model

# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    predictions = scaler.inverse_transform(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# load the new file
dataset = read_csv('b1.csv', header=0, infer_datetime_format=True, parse_dates=['date'], index_col=['date'])
# split into train and test
train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 60
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.title("Line Plot of RMSE per Day for Univariate LSTM with Vector Output and 60-day Inputs")
pyplot.xlabel("Date")
pyplot.ylabel("RMSE")
pyplot.show()
