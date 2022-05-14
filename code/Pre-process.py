# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:29:22 2020

@author: acn980
"""
import os
import scipy as sp
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD

# FUNCTIONS



# LOAD DATA
fn_data = r'E:\github\Coastal-hydrographs\MachineLearning\DATA'
file = 'Merged_Utide_vung_tau_a-383a-vietnam-uhslc.csv'
date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
Xs = pd.read_csv(os.path.join(fn_data, file), parse_dates = True, date_parser= date_parser, index_col = 0, dtype={'msl':np.float32,'u10':np.float32,'v10':np.float32})
Xs = Xs[Xs.columns[-3:]].copy()
Xs['gradient'] = Xs.loc[:,'msl'].diff(1)
Xs['wind'] = Xs['u10']**2 + Xs['v10']**2

fn_res = r'E:\surfdrive\Documents\VU\GESLA\public_11092018_UTide'
file_res = 'vung_tau_a-383a-vietnam-uhslc.csv'
date_parser = lambda x: pd.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
Y = pd.read_csv(os.path.join(fn_res, file_res), parse_dates = True, date_parser= date_parser, index_col = 0)

alls = pd.concat([Xs,Y], axis = 1, join='inner')

#Select only certain time

plt.figure()
plt.plot(alls['wind'], alls['residual'], '.k')
plt.show()
#Link to the tutorial: 
#https://www.kaggle.com/charel/learn-by-example-rnn-lstm-gru-time-series
#https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru/notebook

Enrol_window = 100 #To capture the recurrence
sc = MinMaxScaler(feature_range=(0,1)) #Box-Cox transformation? Only positive // Yeo-Johnson transform?


def normalise_windows(window_data):
    # A support function to normalize a dataset over a window
    normalised_data = []
    for window in window_data: #If not doing LTS seasonality, see Hewamalage et al. 2019
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def load_data(datasetname, column, seq_len, normalise_window):
    # A support function to help prepare datasets for an RNN/LSTM/GRU
    data = datasetname.loc[:,column]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        #result = sc.fit_transform(result)
        result = normalise_windows(result)

    result = np.array(result)

    #Last 10% is used for validation test, first 90% for training
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]