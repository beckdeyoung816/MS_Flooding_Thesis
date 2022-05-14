# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:28:46 2020

@author: acn980
"""

import os
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from tcn import TCN, tcn_full_summary


def series_to_supervised(data, n_in=1, n_out=1):
    """
    data: Sequence of observations as a list or 2D NumPy array. Required.
    n_in: Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1.
    n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
    dropnan: Boolean whether or not to drop rows with NaN values. Optional. Defaults to True.
    
    return: Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
#    # drop rows with NaN values
#    if dropnan:
#        agg.dropna(inplace=True)
        
    return agg

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

#%%
# LOAD DATA
fn_data = r'E:\github\Coastal-hydrographs\MachineLearning\DATA'
file = 'merged_hoekvanholla-hvh-nl-rws.csv'
date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
Xs = pd.read_csv(os.path.join(fn_data, file), parse_dates = True, date_parser= date_parser, index_col = 0)

#Computing gradient and wind force
Xs['gradient'] = Xs.loc[:,'msl'].diff(1)
Xs['wind'] = Xs['u10']**2 + Xs['v10']**2
Xs['u2'] = Xs['u10']**2 
#Xs['u3'] = Xs['u10']**3
Xs['v2'] = Xs['v10']**2 
#Xs['v3'] = Xs['v10']**3
Xs['r_wind'] = [cart2pol(Xs.loc[i,'u10'], Xs.loc[i,'v10'])[0] for i in Xs.index]
Xs['phi_wind'] = [cart2pol(Xs.loc[i,'u10'], Xs.loc[i,'v10'])[1] for i in Xs.index]


#Removing spurious values
Xs.loc[Xs.residual<-5,'residual'] = np.nan
Xs.loc[:,'residual_rolling'] = Xs.loc[:,'residual'].rolling(12,center=True).mean()

#How do you handle NaNs?
Xs.dropna(inplace = True)

#Plotting
fig, axs = plt.subplots(5, 1, sharex=True)
axs[0].plot(Xs.index, Xs.wind, '.-k')
axs[1].plot(Xs.index, Xs.msl, '.-r')
axs[2].plot(Xs.index, Xs.residual, '.-b', Xs.index, Xs.residual_rolling, '.-g')
axs[3].plot(Xs.index, Xs.phi_wind, '-k')
axs[4].plot(Xs.index, Xs.r_wind, '-k')
plt.show()   

#residual = Xs.residual.copy()
#roll_residual = residual.rolling(12,center=True).mean()
#
#plt.figure()
#plt.plot(residual.index, residual, '.-k')
#plt.plot(roll_residual.index, roll_residual, '.-r')
#plt.show()


#We select the predictors and predictand variables
predictors = ['msl', 'gradient', 'phi_wind', 'wind']
predictand = ['residual_rolling']

#Making into np.array()
X = Xs.loc[:,predictors].copy()
Y = Xs[predictand].copy()

#sc = MinMaxScaler(feature_range=(0,1)) #Box-Cox transformation? Only positive // Yeo-Johnson transform?
sc = PowerTransformer(method='yeo-johnson')
X_scaled = sc.fit_transform(X.values)

#sc_y = MinMaxScaler(feature_range=(0,1)) 
sc_y = PowerTransformer(method='yeo-johnson')
Y_scaled = sc_y.fit_transform(Y.values)
#Y_back_scaled = sc_y.inverse_transform(Y_scaled)

#fig, axs = plt.subplots(3, 1, sharex=True)
#axs[0].plot(Xs.residual.values, '-r')
#axs[1].plot(Y_back_scaled, '-r')
#axs[2].plot(Y.values, '-g')
#plt.show() 
 
# frame as supervised learning
scaled = np.concatenate((Y_scaled, X_scaled), axis = 1)
n_before = 1#3*24
n_after = 1
reframed = series_to_supervised(scaled, n_before, n_after) #1 before and 1 after
reframed_index = Xs.index[reframed.index]
reframed.set_index(reframed_index, inplace = True)
reframed.dropna(inplace = True)

reframed = reframed[reframed.columns.drop(list(reframed.filter(regex="var1\(t-")))]
nb_columns = len(reframed.columns)
# drop columns we don't want to predict
col_to_drop = [nb_columns-len(predictors)+i for i in range(len(predictors))]
reframed.drop(reframed.columns[[col_to_drop]], axis=1, inplace=True) 
print(reframed.head())

#%%
# split into train, dev-test, and test sets 
values = reframed.values

train_perc = 0.7
dev_test_perc = 0.15
test_perc = 0.15

n_train_hours = int(np.rint(train_perc * len(reframed))) # 365 * 24 #Fit on first year of data
n_dev_test = int(np.rint(dev_test_perc * len(reframed)))
n_test = int(len(reframed) - n_dev_test - n_train_hours)

train = values[:n_train_hours, :]
test = values[n_train_hours:n_train_hours+n_dev_test, :] #Test on the rest of the dev test data
test_final = values[n_train_hours+n_dev_test:, :]

test_final_X, test_final_y = test_final[:, :-1], test_final[:, -1]
test_final_X = test_final_X.reshape((test_final_X.shape[0], n_before, len(predictors)))

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_index = reframed.index[:n_train_hours]
test_index = reframed.index[n_train_hours:n_train_hours+n_dev_test]
test_final_index = reframed.index[n_train_hours+n_dev_test:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_before, len(predictors)))
test_X = test_X.reshape((test_X.shape[0], n_before, len(predictors)))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#%% https://github.com/philipperemy/keras-tcn/blob/master/README.md
batch_size, timesteps, input_dim = None, n_before, len(predictors)

def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train


i = Input(batch_shape=(batch_size, n_before, input_dim))

o = TCN(nb_filters=100, dropout_rate=0.2, return_sequences=True)(i)
o = TCN(nb_filters=100, dropout_rate=0.2, return_sequences=False)(o)  # The TCN layers are here.
o = Dense(1)(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')

tcn_full_summary(m, expand_residual_blocks=False)

m.fit(train_X, train_y, epochs=10, validation_split=0.2)

#%% https://github.com/philipperemy/keras-tcn/blob/master/README.md
#PREDICTION
# make a prediction
yhat = m.predict(test_X)

# invert scaling for forecast
inv_yhat = sc_y.inverse_transform(yhat)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = sc_y.inverse_transform(test_y)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.figure()
plt.plot(test_y, '-r')
plt.plot(yhat, '-k')
plt.plot()

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(test_index, inv_y, '-k')
axs[1].plot(test_index, inv_yhat, '-r')
axs[2].plot(Xs.index, Xs.residual, '.-k', Y.index, Y, '-r')
plt.show()