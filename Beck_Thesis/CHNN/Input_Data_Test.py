# %%
import logging
import logging.handlers
import os
import sys
from matplotlib import ticker
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import time
import datetime
import xarray as xr
from keras import utils, layers
from keras.models import Sequential
import tensorflow_probability as tfp
# import darts
# from darts.models import TCNModel
# from darts import datasets as ds
import tcn


from LSTM import LSTM
import to_learning
import performance
# import model_run_coast as mr
import Coastal_Model as cm

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir('/Users/beck/My Drive/VU/Thesis/Scripts/Beck_Thesis/')


# %%
def set_logger(arg_start, arg_end, verbose=True):
    """
    Set-up the logging system, exit if this fails
    """
    # assign logger file name and output directory
    datelog = time.ctime()
    datelog = datelog.replace(':', '_')
    reference = f'ML_stormsurges_loop_{arg_start}-{arg_end}'


    logfilename = ('logger' + os.sep + reference + '_logfile_' + 
                   str(datelog.replace(' ', '_')) + '.log')

    # create output directory if not exists
    if not os.path.exists('logger'):
        os.makedirs('logger')

    # create logger and set threshold level, report error if fails
    try:
        logger = logging.getLogger(reference)
        logger.setLevel(logging.DEBUG)
    except IOError:
        sys.exit('IOERROR: Failed to initialize logger with: ' + logfilename)

    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                  '%(levelname)s - %(message)s')

    # assign logging handler to report to .log file
    ch = logging.handlers.RotatingFileHandler(logfilename,
                                              maxBytes=10*1024*1024,
                                              backupCount=5)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # assign logging handler to report to terminal
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # start up log message
    logger.info('File logging to ' + logfilename)

    return logger, ch

complexity=False
year='last'
fn_exp='Models'
arg_count=0
mask_val=-999
hyper_opt=False
NaN_threshold=0

station = 'cuxhaven-cuxhaven-germany-bsh'  # 'Cuxhaven' 'Hoek van Holland', Puerto Armuelles
# station = 'harvest_oil_p.,ca-594a-usa-uhslc'
resample = 'hourly' # 'hourly' 'daily'
resample_method = 'rolling_mean'  # 'max' 'res_max' 'rolling_mean' ## res_max for daily and rolling_mean for hourly
variables = ['msl', 'grad', 'u10', 'v10', 'rho', 'sst']  # 'grad', 'rho', 'phi', 'u10', 'v10', 'uquad', 'vquad'
tt_value = 0.67  # train-test value
scaler_type = 'std_normal'  # std_normal, MinMax
n_ncells = 0
epochs = 25
batch = 100
batch_normalization = False
neurons = 48
filters = 8
n_layers = 1  # now only works for uniform layers with same settings
activation = 'relu'  # 'relu', 'swish', 'Leaky ReLu', 'sigmoid', 'tanh'
loss = 'mae' #'test' #'Gumbel' #'mae'  # 'mae', 'mean_squared_logarithmic_error', 'mean_squared_error'
optimizer = tf.keras.optimizers.Adam(clipnorm=1) # 'adam'  # SGD(lr=0.01, momentum=0.9), 'adam'
dropout = True
drop_value = 0.2
l1, l2 = 0, 0.01

ML = 'ANN'  # 'LSTM', 'CNN', 'ConvLSTM', 'ANN', 'ALL', 'LSTM_TCN'
model_dir = os.path.join(os.getcwd(), 'Models')
name_model = '{}_surge_ERA5'.format(ML)
input_dir = 'Input_nc_sst'
output_dir = 'ML_model'
figures_dir = 'Figures'
year = 'last'
frac_ensemble = 0.5

loop = 2
frac_ens=frac_ensemble
verbose=0
validation='select'
logger, ch = set_logger(loop, n_ncells)
earlyStoppingMonitor = EarlyStopping(patience=3, restore_best_weights = 'True')

# %%
def get_input_data(station, variables, ML, input_dir, resample, resample_method, batch,
                   scaler_type, year, n_ncells, 
                   mask_val, logger):
    ML_list = [ML]
    print('ML_list is:', ML_list)

    df, lat_list, lon_list, direction, scaler, reframed, test_dates, i_test_dates = to_learning.prepare_station(station, variables, ML, input_dir, resample, resample_method,
                                                                                                                cluster_years=5, extreme_thr=0.02, sample=False, make_univariate=False,
                                                                                                                scaler_type=scaler_type, year = year, scaler_op=True, n_ncells=n_ncells, mask_val=mask_val, logger=logger)
    if resample == 'hourly':                            
        batch = batch * 24

    # split testing phase year    
    test_year = reframed.iloc[i_test_dates].copy()

    #NaN masking the complete test year 
    reframed.iloc[i_test_dates] = np.nan  

    # reframe ML station data
    test_year.loc[test_year.iloc[:,-1].isna(),'values(t)'] = mask_val #Changing all NaN values in residual testing year to masking_val                                                                                                                                            
    _, _, test_X, test_y, _ = to_learning.splitting_learning(test_year, df, 0, ML, variables, direction, lat_list, lon_list, batch, n_train=False)

    result_all = dict()
    result_all['data'] = dict()
    result_all['train_loss'] = dict()
    result_all['test_loss'] = dict()
    for i in range(loop):
        if not logger:
            print(f'\nEnsemble loop: {i + 1}\n')
        tf.keras.backend.clear_session()
        name_model = f'{ML}_ensemble_{i + 1}_{loss}'

        # shuffle df
        reframed_ensemble = reframed.copy()
        reframed_draw, n_train = to_learning.select_ensemble(reframed_ensemble, 'values(t)', ML, batch, tt_value=tt_value, frac_ens = frac_ens, mask_val=mask_val, NaN_threshold=NaN_threshold) 

        # We modify the input data so that it is masked        
        reframed_draw = reframed_draw.reset_index(drop=True).copy()
        reframed_draw[reframed_draw.iloc[:,-1]==mask_val] = mask_val
        # print('There are so many Nan: ', sum(reframed_draw.iloc[:,-1]==mask_val))
        
        # reframe ML station data
        train_X, train_y, val_X, val_y, n_train = to_learning.splitting_learning(reframed_draw, df, tt_value, ML, variables, direction, lat_list, lon_list, batch, n_train=n_train)
        
        # Hyperparameter optimization
        if hyper_opt:
            sherpa_output = os.path.join('Hyper_opt', 'complexity', station, ML)
            if not os.path.exists(sherpa_output):
                os.makedirs(sherpa_output)
            #lstm.hyper_opt()
            continue
        else:
            sherpa_output = None
    return test_X, test_y, train_X, train_y, val_X, val_y, n_train, result_all, sherpa_output, df

# def gumbel_loss_hyper(gamma=1.1):
#     def gumbel_loss(y_true, y_pred):
#         u = y_pred - y_true
        
#         a = 1 - K.exp(-K.pow(u, 2))
#         b= K.pow(a, gamma)
#         c = tf.multiply(b, K.pow(u,2))
#         d =K.exp(-c)
#         e = K.mean(d)

#         ll = -K.log(e)
#         return ll
    
#     return gumbel_loss

def gumbel_loss_hyper(gamma=1.1):
    def gumbel_loss(y_true, y_pred):
        u = y_pred - y_true
        
        a = 1 - K.exp(-K.pow(u, 2))
        b= K.pow(a, gamma)
        c = tf.multiply(b, K.pow(u,2))
        
        return K.mean(c)
    
    return gumbel_loss


def frechet_loss_hyper(alpha, s):
    def frechet_loss(y_true, y_pred):
        
        # tf.print('\nTrue: ', y_true)
        # tf.print('\nPred: ', y_pred)
        u = y_pred - y_true
        # tf.print('\nu: ', u)
        a = (u + s*((alpha/(1+alpha)) ** (1/alpha))) / s
        # tf.print('\na: ', a)
        b = K.pow(tf.multiply(-1-alpha, -a), -alpha) + K.log(a)
        # tf.print('\nb: ', b)
        c = tf.where(u < 0.0, 100.0, b)
        loss = K.mean(tf.where(y_pred < 0.0, 1000.0, c))
        #tf.print('\n\nLoss: ', loss)
        return loss
    return frechet_loss


# def frechet_loss_hyper(alpha, s):
#     def frechet_loss(y_true, y_pred):
#         u = y_pred - y_true
        
#         a = (alpha/(1+alpha)) ** (1/alpha)
#         print(f'{a=:.2f}')
#         b = (u + (s * a)) / s 
#         tf.print(b)
        
#         # b = (u + s*(alpha/(1+alpha) ** (1/alpha))) / s
        
#         c = K.pow(tf.multiply(-1-alpha, -b), -alpha) + K.log(b)
        
#         return K.mean(tf.where(u < 0, 0.0, c))
    
#     return frechet_loss



# def frechet_loss(y_true, y_pred):
#     alpha = 13.0
#     s = 1.7
#     u = y_pred - y_true
#     #tf.print('\nu:', u)
    
#     b = (u + s*(alpha/(1+alpha) ** (1/alpha))) / s
#     # #a = K.pow(tf.divide(alpha, 1+alpha),tf.divide(1,alpha))
#     # tf.print('\na:', a)
#     # ab = u + tf.multiply(s, a)
#     # tf.print('\nab:', ab)
#     # b = tf.divide(ab,s)
#     ##print('\nb:', b)
#     #a = (u + s*(alpha/(1+alpha) ** (1/alpha))) / s
#     c = K.pow(tf.multiply(-1-alpha, -b), -alpha) + K.log(b)
#     # tf.print('\nc:', c)
#     #c = (-1-alpha) * (-b) ** (-alpha) + K.log(b)
    
#     d = K.mean(tf.where(u < 0, 0.0, c))

#     ll = -K.log(d)
    
#     return ll

def plot_res(model, test_X, test_y):
    pred = model.predict(test_X)
    pred = [x[0] for x in pred]
    plt.plot(test_y, color='blue', alpha=.5)
    plt.plot(pred, color='red', alpha=.5)
    plt.legend(['True', 'Pred'])
    plt.show()
    
    return pred

# %%
ann_test_X, ann_test_y, ann_train_X, ann_train_y, ann_val_X, ann_val_y, ann_n_train, ann_result_all, ann_sherpa_output, df = get_input_data(station, variables, 'ANN', input_dir, resample, resample_method, batch,
                                                                                                                                        scaler_type, year, n_ncells, mask_val, logger)
# %%
loss_fun = gumbel_loss_hyper(tf.Variable([1.1]))
# loss_fun = frechet_loss_hyper(alpha=tf.Variable([13.0]), s=tf.Variable([1.7]))
#loss_fun = frechet_loss_hyper(alpha=10.0, s=5.1)

act = 'relu'

ann_model = Sequential()
ann_model.add(layers.Dense(15, activation=act, input_dim=ann_train_X.shape[1], kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)))
ann_model.add(layers.Dense(10, activation=act, kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)))
# model.add(layers.Dense(5, activation='tanh'))
ann_model.add(layers.Dense(1))
# model.add(layers.Dense(5, activation='relu'))

ann_model.compile(optimizer=keras.optimizers.Adam(clipnorm=1), 
                  loss=loss_fun, 
                  metrics=['mae', 'mse'])

# %%

ann_history = ann_model.fit(ann_train_X, ann_train_y, epochs=20, batch_size=2400, validation_data=(ann_val_X, ann_val_y),
                            callbacks = None, shuffle=False)
# %%
plot_res(ann_model, ann_test_X, ann_test_y)

plt.plot(ann_history.history['val_loss'])

# %%
ann_model.fit_generator()


# %%
lstm_test_X, lstm_test_y, lstm_train_X, lstm_train_y, lstm_val_X, lstm_val_y, lstm_n_train, lstm_result_all, lstm_sherpa_output, df = get_input_data(station, variables, 'LSTM', input_dir, resample, resample_method, batch,
                                                                                                                                        scaler_type, year, n_ncells, 
                                                                                                                                        mask_val, logger)
input_shape = (lstm_train_X.shape[1], lstm_train_X.shape[2])
# %%

lstm_model = Sequential()
lstm_model.add(layers.LSTM(48, input_shape=input_shape, activation='relu', return_sequences=True))
lstm_model.add(layers.Dropout(.2))
lstm_model.add(layers.LSTM(24, activation='relu', return_sequences=False))
lstm_model.add(layers.Dropout(.2))
lstm_model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(5, activation='tanh'))
lstm_model.add(layers.Dense(1))

# %%
lstm_model.compile(optimizer='adam', loss=loss_fun, metrics=['mae', 'mse',])
lstm_history = lstm_model.fit(lstm_train_X, lstm_train_y, epochs=20, batch_size=2400,
                         #workers=3, use_multiprocessing=True,
                         validation_data=(lstm_val_X, lstm_val_y),
                                callbacks = [earlyStoppingMonitor], shuffle=False)

# %%
plot_res(lstm_model, lstm_test_X, lstm_test_y)

plt.plot(lstm_history.history['val_loss'])

# %%
lstm_df = pd.DataFrame(lstm_train_X[:,0,:])
ann_df = pd.DataFrame(ann_train_X)


# %%
y_true = tf.constant([[x] for x in ann_val_y[0:5]])
y_pred = ann_model.predict(ann_val_X[0:5])


# %%
y_pred = np.array([5.0,1.0,3.5,2.0,0.0], dtype=np.float64)
y_true = np.array([5.0,1.0,3.5,2.0,20.0], dtype=np.float64)

u = y_pred - y_true

alpha = tf.Variable(13.0, dtype=np.float64)
s = tf.Variable(1.7, dtype=np.float64)

b = (u + s*(alpha/(1+alpha) ** (1/alpha))) / s
c = K.pow(tf.multiply(-1.0-alpha, -b), -alpha) + K.log(b)
d = K.mean(tf.where(u<0, 0.0, c))

print(d.numpy())
# %%

# %%
# TCN TESTING


# # %%
# tcn_train_X = darts.TimeSeries.from_values(ann_train_X)
# tcn_train_y = darts.TimeSeries.from_values(ann_train_y)

# tcn_test_X = darts.TimeSeries.from_values(ann_test_X)
# tcn_test_y = darts.TimeSeries.from_values(ann_test_y)

# # %%
# tcn_model = TCNModel(
#     input_chunk_length=24,
#     output_chunk_length=6,
#     n_epochs=10,
#     dropout=0.1,
#     dilation_base=2,
#     kernel_size=5,
#     num_filters=3,
#     random_state=0
# )

# tcn_model.fit(series = tcn_train_y, past_covariates = tcn_train_X)
# # %%
# tcn_backtest = tcn_model.historical_forecasts(
#     series=tcn_test_y,
#     past_covariates=tcn_test_X,
#     retrain=False,
#     verbose=False
# )

# # %%
# plt.figure(figsize=(10, 6))
# tcn_test_y.plot(label="actual")
# tcn_backtest.plot(label="backtest")
# plt.legend()

# %%
tcn_train_X, tcn_train_y = lstm_train_X, lstm_train_y
tcn_test_X, tcn_test_y = lstm_test_X, lstm_test_y
tcn_val_X, tcn_val_y = lstm_val_X, lstm_val_y
# %%
tcn_model = Sequential()
tcn_model.add(tcn.TCN(48, input_shape=input_shape, activation='relu', return_sequences=True,
              dilations=(1,2,4,8), dropout_rate=.2, kernel_size=3))
tcn_model.add(layers.LSTM(24, activation='relu', return_sequences=False))
# tcn_model.add(layers.Dropout(.2))
tcn_model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(5, activation='tanh'))
tcn_model.add(layers.Dense(1, activation = 'softplus'))
# %%
tcn_model.compile(optimizer='adam', loss=loss_fun, metrics=['mae', 'mse',])
tcn_history = tcn_model.fit(tcn_train_X, tcn_train_y, epochs=20, batch_size=2400,
                         #workers=3, use_multiprocessing=True,
                         validation_data=(tcn_val_X, tcn_val_y),
                                callbacks = [earlyStoppingMonitor], shuffle=False)
# %%
tcn_model.evaluate(tcn_test_X, tcn_test_y)
tcn_pred = tcn_model.predict(tcn_test_X)
tcn_pred = [x[0] for x in tcn_pred]
plt.plot(tcn_test_y, color='blue', alpha=.5)
plt.plot(tcn_pred, color='red', alpha=.5)
# %%
utils.plot_model(
    tcn_model,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    dpi=200,
    layer_range=None,
)

# %%
ML_list = [ML]
print('ML_list is:', ML_list)

df, lat_list, lon_list, direction, scaler, reframed, test_dates, i_test_dates = to_learning.prepare_station(station, variables, ML, input_dir, resample, resample_method,
                                                                                                            cluster_years=5, extreme_thr=0.02, sample=False, make_univariate=False,
                                                                                                            scaler_type=scaler_type, year = year, scaler_op=True, n_ncells=n_ncells, mask_val=mask_val, logger=logger)
# %%
if resample == 'hourly':                            
    batch = batch * 24

# split testing phase year    
test_year = reframed.iloc[i_test_dates].copy()

#NaN masking the complete test year 
reframed.iloc[i_test_dates] = np.nan  

# reframe ML station data
test_year.loc[test_year.iloc[:,-1].isna(),'values(t)'] = mask_val #Changing all NaN values in residual testing year to masking_val                                                                                                                                            
_, _, test_X, test_y, _ = to_learning.splitting_learning(test_year, df, 0, ML, variables, direction, lat_list, lon_list, batch, n_train=False)


# %%
df, ds, direction = to_learning.load_file(station, input_dir)
# %%
df2, lat_list, lon_list = to_learning.spatial_to_column(df, ds, variables, [], n_ncells)
# %%
res = pd.DataFrame([ann_test_y, ann_pred], index = ['Obs', 'Pred']).T
# %%
res['Obs_Ext'] = res['Obs'] > 1.75
res['Pred_Ext'] = res['Pred'] > 1.75
res['Wrong'] = res['Obs_Ext'] != res['Pred_Ext']
res['Residual'] = res['Obs'] - res['Pred']
# %%
from sklearn.metrics import confusion_matrix, mean_squared_error as mse
# %%

# %%
tn, fp, fn, tp = confusion_matrix(res['Obs_Ext'], res['Pred_Ext'], labels=[0,1]).ravel()
# %%
(tn,fp,fn,tp)
# %%
np.sqrt(mse(res['Obs'], res['Pred']))
np.sqrt(mse(res['Obs'][res['Obs_Ext']], res['Pred'][res['Obs_Ext']]))
np.sqrt(mse(res['Obs'][not res['Obs_Ext']], res['Pred'][not res['Obs_Ext']]))
# %%
data = np.linspace(-2, 2, 100)

def gumbel_kde(u, gamma):
    return np.exp((-(1-np.exp(-u**2)) ** gamma) * (u **2))
Z = 1
plt.plot(Z*gumbel_kde(data, gamma=1), label = 'Gamma = 1' )
plt.plot(Z*gumbel_kde(data, gamma=1.1), label = 'Gamma = 1.1' )
plt.plot(Z*gumbel_kde(data, gamma=1.5), label = 'Gamma = 1.5' )
plt.plot(Z*gumbel_kde(data, gamma=2), label = 'Gamma = 2' )
plt.legend()

plt.show()

# %%
data = np.linspace(-2, 2, 1000)

def frechet_kde(u, s, alpha): 
    a = (alpha/(1+alpha)) ** (1/alpha)
    b = (u + s*a) / s 
    c = (alpha/s) * (b ** (-1-alpha))
    d = np.exp(-(b ** (-alpha)))
        
    return c * d

Z = .1
plt.plot(data, Z * frechet_kde(data, alpha=10, s = 1.7), label = 'A=10 ; s = 1.7')
plt.plot(data, Z * frechet_kde(data, alpha=13, s = 1.7), label = 'A=13 ; s = 1.7')
plt.plot(data, Z * frechet_kde(data, alpha=15, s = 1.7), label = 'A=15 ; s = 1.7')
plt.plot(data, Z * frechet_kde(data, alpha=15, s = 2.0), label = 'A=15 ; s = 2.0')
plt.ylim(0,.4)
plt.legend()
plt.show()

# %%
def frechet_loss_hyper2(alpha, s, y_true, y_pred):
        u = y_pred - y_true
        # tf.print('\nu: ', u)
        a = K.pow((alpha/(1+alpha)), (1/alpha))
        b = tf.divide((u + (s * a)), s)
        b2 =  K.pow(-b, (-alpha))
        c = tf.multiply((-1 - alpha), b2)
        d = K.log(b)
        
        e = tf.where(u < 0.0, 10.0, c+d)
        
        return tf.where(y_pred < 0.0, 10.0, e)
        
        # a = (u + s*((alpha/(1+alpha)) ** (1/alpha))) / s
        # # tf.print('\na: ', a)
        # b = K.pow(tf.multiply(-1-alpha, -a), -alpha) + K.log(a)
        # # tf.print('\nb: ', b)

        # # loss = K.mean(tf.where(u < 0.0, 0.0, b))
        # # tf.print('\n\nLoss: ', loss)
        # return tf.where(u < 0.0, 0.0, b)
    
    
y_pred = np.linspace(-20.,20.,5000)
y_true = tf.constant(0.)

Z = 1

# %%
A1, s1 = 10, 10.0
A2, s2 = 13, 11
A3, s3 = 15, 1.7
A4, s4 = 15, 2.0

fig, axes = plt.subplots(2, 2, figsize=(10,8))
axes = axes.ravel()
axes[0].plot(y_pred, Z * frechet_loss_hyper2(y_pred = y_pred, y_true=y_true, alpha=A1, s = s1))
axes[0].set_title(f'A={A1} ; s = {s1}')
axes[1].plot(y_pred, Z * frechet_loss_hyper2(y_pred = y_pred, y_true=y_true, alpha=A2, s = s2))
axes[1].set_title(f'A={A2} ; s = {s2}')
axes[2].plot(y_pred, Z * frechet_loss_hyper2(y_pred = y_pred, y_true=y_true, alpha=A3, s = s3))
axes[2].set_title(f'A={A3} ; s = {s3}')
axes[3].plot(y_pred, Z * frechet_loss_hyper2(y_pred = y_pred, y_true=y_true, alpha=A4, s = s4))
axes[3].set_title(f'A={A4} ; s = {s4}')
plt.show()
# %%
def gumbel_loss_hyper2(gamma, y_true, y_pred):
        u = y_pred - y_true
        
        a = 1 - K.exp(-K.pow(u, 2))
        b= K.pow(a, gamma)
        c = tf.multiply(b, K.pow(u,2))
        return c
        return K.mean(c)
    
    
    
y_pred = np.linspace(-1000.,1000.,5000)
y_true = tf.constant(0.)

val1 = gumbel_loss_hyper2(1.1, y_true,y_pred)

plt.plot(y_pred, val1)
# %%
# %%
coast_stations = ['calais-calais-france-refmar',         
                'denhelder-hel-nl-rws',
                'aberdeen-p038-uk-bodc',
                'cuxhaven-cuxhaven-germany-bsh',
                'esbjerg-130121-denmark-dmi',
                'brest-brest-france-refmar',
                'delfzijl-del-nl-rws',
                'hoekvanholla-hvh-nl-rws']


act = 'relu'
loss_fun = 'mse'
ann_model = Sequential()
ann_model.add(layers.Dense(15, activation=act, input_dim=len(variables), kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)))
ann_model.add(layers.Dense(10, activation=act, kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)))
ann_model.add(layers.Dense(1))

ann_model.compile(optimizer=keras.optimizers.Adam(clipnorm=1), 
                  loss=loss_fun, 
                  metrics=['mae', 'mse'])

# %%
ann_history = {}
ann_trains_X = {}
ann_tests_X = {}
ann_trains_y = {}
ann_tests_y = {}

num_stations = 4

for station in coast_stations[0:num_stations]:
    ann_test_X, ann_test_y, ann_train_X, ann_train_y, ann_val_X, ann_val_y, ann_n_train, ann_result_all, ann_sherpa_output, df = get_input_data(station, variables, 'ANN', input_dir, resample, resample_method, batch,
                                                                                                                                        scaler_type, year, n_ncells, mask_val, logger)
    
    ann_history[station] = ann_model.fit(ann_train_X, ann_train_y, epochs=20, batch_size=2400, validation_data=(ann_val_X, ann_val_y),
                            callbacks = None, shuffle=False)
    
    ann_trains_X[station] = ann_train_X
    ann_tests_y[station] = ann_test_y
    ann_trains_y[station] = ann_train_y
    ann_tests_X[station] = ann_test_X
    
# %%
ann_preds = {}
mses = np.array([])
for station in coast_stations[0:num_stations]:
    ann_preds[station] = plot_res(ann_model, ann_tests_X[station], ann_tests_y[station])
    mses = np.append(mses, ann_model.evaluate(ann_tests_X[station], ann_tests_y[station])[2])
    plt.plot(ann_history[station].history['val_loss'], label = station)
    plt.legend()
    plt.show()
# %%
mses.mean()
# %%
ann_preds = {}
mses = np.array([])

fig, axes = plt.subplots(2, num_stations // 2 + (num_stations % 2), figsize=(12,8))
axes = axes.ravel()

for i, station in enumerate(coast_stations[0:num_stations]):
    pred = ann_model.predict(ann_tests_X[station])
    pred = [x[0] for x in pred]
    mse = ann_model.evaluate(ann_tests_X[station], ann_tests_y[station])[2]
    mses = np.append(mses, mse)
    
    axes[i-1].plot(ann_tests_y[station], color='blue', alpha=.5)
    axes[i-1].plot(pred, color='red', alpha=.5)
    axes[i-1].set_title(f'{station} ; RMSE = {np.sqrt(mse):.2f}')
    plt.legend(['True', 'Pred'])
    
plt.show()
# %%
# station = mrb.get_input_data(coast_stations[0], variables, ML, input_dir, resample, resample_method, batch,
#                    scaler_type, year, n_ncells, mask_val, tt_value, frac_ens, NaN_threshold,
#                    logger)
# %%
coast = 'NE Atlantic Yellow'
stations = {}
for station in mr.get_coast_stations(coast):
    # Get input data for the station
    # This includes the train, test, and validation data, as well as the scaler and transformed data for inverse transforming
    stations[station] = mr.get_input_data(station, variables, ML, input_dir, resample, resample_method, batch, scaler_type, year, n_ncells, mask_val, tt_value, frac_ens, NaN_threshold, logger)
                                
# %%
sherpa_output = None
verbose = 1
loss = 'gumbel'
ML = 'ANN'
model = cm.Coastal_Model(stations, ML, loss, n_layers, neurons, activation, dropout, drop_value,
                                      hyper_opt, validation, optimizer, epochs, batch, verbose, model_dir, filters,
                                      variables, batch_normalization, sherpa_output, logger, name_model,
                                      alpha=None, s=None, gamma=1.1, l1=l1, l2=l2, mask_val=mask_val)

# %%
i = 0
model.design_network()
model.compile()
model.train_model(i=i)
# %%
station = model.station_inputs['calais-calais-france-refmar']
# for station in model.station_inputs.values():
#             station.predict(self.model, ensemble_loop, self.mask_val)
# %%
temp_df = station.test_year.replace(to_replace=mask_val, value=np.nan)[station.n_train_final:].copy()

# %%
# make a prediction
station.test_preds = model.model.predict(station.test_X)

# %%

# invert scaling for observed surge
station.inv_test_y = station.scaler.inverse_transform(temp_df.values)[:,-1]

# %%
# invert scaling for modelled surge
temp_df.loc[:,'values(t)'] = station.test_preds
station.inv_test_preds = station.scaler.inverse_transform(temp_df.values)[:,-1]

# %%
station.rmse = np.sqrt(mean_squared_error(station.inv_test_y, station.inv_test_preds))

# %%
# Store Results
df_all = performance.store_result(station.inv_test_preds, station.inv_test_y)
df_all = df_all.set_index(station.df.iloc[station.test_dates,:].index, drop = True)                                                                    

station.result_all['data'][ensemble_loop] = df_all.copy()


# %%
for station in stations.values():
    plt.hist(station.train_y)
    plt.title(station.name)
    plt.show()
# %%
