""" 
This script contains the Coastal_Model class
It contains methods such as designing a network for a TCN, LSTM, and ANN model.
Then compiling and training the model.

"""

import keras
from keras import layers
from keras import models
from keras.backend import sigmoid
from keras import utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ProgbarLogger
import os

import sherpa
import tensorflow as tf
from tensorflow import math as tfm
import numpy as np
import random
import keras.backend as K
import tcn

def reset_seeds():
    #Solution to reset random states from: https://stackoverflow.com/questions/58453793/the-clear-session-method-of-keras-backend-does-not-clean-up-the-fitting-data 
    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    print("RANDOM SEEDS RESET")


def swish_func(x, beta = 1):
    return (x * sigmoid(beta * x))


def relu_advanced_func(x, threshold = -3):
    return keras.activations.relu(x, threshold=threshold)


class swish(layers.Activation):
    def __init__(self, activation, **kwargs):
        super(swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


class relu_advanced(layers.Activation):
    def __init__(self, activation, **kwargs):
        super(relu_advanced, self).__init__(activation, **kwargs)
        self.__name__ = 'relu_advanced'

class Coastal_Model():
    def __init__(self, station_inputs, ML, loss, n_layers, neurons, activation, dropout, drop_value, 
                 hyper_opt, validation, optimizer, epochs, batch, verbose, model_dir, filters, 
                 variables, batch_normalization, sherpa_output, logger, name_model,
                 alpha=13, s=1.7, gamma=1.1, l1=0.01, l2=0.01, mask_val=-999):
        
        # Model parameters
        self.ML = ML
        self.n_layers = n_layers
        self.neurons = neurons
        self.activ = activation
        self.drop_out = dropout
        self.drop_value = drop_value
        self.l1 = l1
        self.l2 = l2
        self.mask_val = mask_val
        self.hyp_opt = hyper_opt
        self.validation = validation
        self.filters = filters
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        
        # Loss function parameters
        if loss.lower() == 'gumbel':
            self.gamma = gamma
            self.custom_loss_fn = self.gumbel_loss_hyper(gamma=gamma)
            
        elif loss.lower() == 'frechet':
            self.alpha = alpha
            self.s = s
            self.custom_loss_fn = self.frechet_loss()
        else:
            self.custom_loss_fn = 'mse'
            
        
        # Data & Misc
        self.station_inputs = station_inputs
        self.batch_size = batch
        self.model_dir = model_dir
        self.name_model = name_model
        self.batch_norm = batch_normalization
        self.vars = variables
        self.sherpa_output = sherpa_output
        self.logger = logger
    
    
    
    def ANN_model(self):
        """
        Design an ANN model
        """

        model = models.Sequential()
        model.add(layers.Dense(self.neurons, kernel_regularizer=keras.regularizers.l1_l2(l1=self.l1, l2=self.l2), activation=self.activ, input_dim=len(self.vars)))
        for i in range(self.n_layers - 1):
            model.add(layers.Dense(self.neurons, kernel_regularizer=keras.regularizers.l1_l2(l1=self.l1, l2=self.l2), activation=self.activ))
            
        if self.drop_out:
            model.add(layers.Dropout(self.drop_value))
            
        model.add(layers.Dense(1))
        self.model = model
    
    def LSTM_model(self):
        """
        Design an LSTM model
        """
        # Time steps, num features
        input_shape = (1, len(self.vars))

        model = models.Sequential()
        for i in range(self.n_layers):
            rs = False if i == self.n_layers - 1 else True # Only return sequences if another LSTM layer is coming
            if i == 0: # Specify the input shape for the first layer
                model.add(layers.LSTM(self.neurons, input_shape=input_shape, activation=self.activ, return_sequences=rs,
                                      name='lstm_input'))
            else:
                model.add(layers.LSTM(self.neurons, activation=self.activ, return_sequences=rs,
                                      name=f'lstm_{i}'))
        model.add(layers.Dropout(self.drop_value, name='dropout_1'))
        model.add(layers.Dense(self.neurons, activation=self.activ, name='dense_1'))
        model.add(layers.Dropout(self.drop_value, name='dropout_2'))
        model.add(layers.Dense(1, name='dense_output'))
        
        self.model = model
        
        
    def TCN_model(self, lstm=False):
        """
        Design a TCN model
        """
        # Time steps, num features
        input_shape = (1, len(self.vars))
        
        model = models.Sequential()
        for i in range(self.n_layers):
            rs = i < self.n_layers-1 or lstm # Only return sequences if another TCN layer is coming and no LSTM layer is added
            if i == 0: # Specify the input shape for the first layer
                model.add(tcn.TCN(self.neurons, input_shape=input_shape, activation='relu', return_sequences=rs,
                    dilations=(1,2,4,8), dropout_rate=self.drop_value, kernel_size=5, name='tcn_input'))
            else:
                model.add(tcn.TCN(self.neurons, activation='relu', return_sequences=rs,
                    dilations=(1,2,4,8), dropout_rate=self.drop_value, kernel_size=5, name=f'tcn_{i}'))
        if lstm:
            model.add(layers.LSTM(self.neurons, activation=self.activ, return_sequences=False))
                
        model.add(layers.Dropout(self.drop_value, name='dropout_1'))
        model.add(layers.Dense(self.neurons, activation=self.activ, name='dense_1'))
        model.add(layers.Dropout(self.drop_value, name='dropout_2'))
        model.add(layers.Dense(1, name='dense_output'))
        
        self.model = model
    
    def design_network(self):
        """
        Design desired network type based on the ML parameter
        """
        
        # design Network
            
        # if self.activ == 'swish':
        #     utils.generic_utils.get_custom_objects().update({'swish': swish(swish_func)})

        # if self.activ == 'relu':
        #     self.activ = 'relu_advanced'
        #     utils.generic_utils.get_custom_objects().update({'relu_advanced': relu_advanced(relu_advanced_func)})
        
        if self.ML == 'ANN':
            self.ANN_model()
        elif self.ML == 'LSTM':
            self.LSTM_model()
        elif self.ML == 'TCN':
            self.TCN_model()
        elif self.ML == 'TCN-LSTM':
            self.TCN_model(lstm=True)
            
            
    def compile(self):
        self.model.compile(loss=self.custom_loss_fn, optimizer=self.optimizer)
        self.model.summary()
        
    
    def gumbel_loss_hyper(self, gamma=1.1):
        def gumbel_loss(y_true, y_pred):
            u = y_pred - y_true
            
            a = 1 - K.exp(-K.pow(u, 2))
            b= K.pow(a, gamma)
            c = tf.multiply(b, K.pow(u,2))
            d =K.exp(-c)
            e = K.mean(d)

            ll = -K.log(e)
            return ll
    
        return gumbel_loss

    def frechet_loss(self):
        def frechet_loss_fn(y_true, y_pred):
            delta = y_pred - y_true
        
            delta_S = (delta + self.s*(self.alpha/(1+self.alpha) ** (1/self.alpha))) / self.s

            loss = (-1-self.alpha) * (-delta_S) ** (-self.alpha) + \
                tfm.log(delta_S)
            
            # return tf.reduce_mean(loss)
            return K.mean(tf.where(delta < 0, 0, loss))
        
        return frechet_loss_fn
    
    def train_model(self, i, hyper_opt=False):
            
        my_callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode="auto", restore_best_weights = 'True'),
                        ModelCheckpoint(filepath=os.path.join(self.model_dir, self.name_model), monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)] # ProgbarLogger(count_mode="steps", stateful_metrics=None), , ModelCheckpoint(filepath=os.path.join(model_dir, name_model), monitor='loss', save_best_only=True, save_weights_only=False, mode='auto', period=1),
        # my_callbacks = [ModelCheckpoint(filepath=os.path.join(model_dir, name_model), monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)]
        
        if hyper_opt:
            my_callbacks.append(self.hyper_opt)

        if self.ML in ['LSTM', 'TCN', 'TCN-LSTM']:
            shuffle = False #'batch'
        else:
            shuffle= True
            
        
        self.history = {}
        
        # Fit network sequentially on each station
        train_stations = [station for station in self.station_inputs.values() if station.train_test == 'Train']
        num_stations = len(train_stations)
        for j, station in enumerate(train_stations):
            print(f'\nTraining Station ({j+1} of {num_stations}): {station.name}\n')
            # fit network
            if self.validation == 'split':
                self.history[station.name] = self.model.fit(station.train_X, station.train_y, epochs=self.epochs, batch_size=self.batch_size, 
                                    validation_split=0.3, callbacks=my_callbacks, verbose=self.verbose, shuffle=shuffle)
            elif self.validation == 'select':
                self.history[station.name] = self.model.fit(station.train_X, station.train_y, epochs=self.epochs, batch_size=self.batch_size,
                                    validation_data=(station.val_X, station.val_y), callbacks=my_callbacks, verbose=self.verbose, shuffle=shuffle)
            else:
                raise ValueError('Validation must be either "split" or "select"')
            
            station.result_all['train_loss'][i] = self.history[station.name].history['loss']
            station.result_all['test_loss'][i] = self.history[station.name].history['val_loss']

        self.model.save(os.path.join(self.model_dir, self.name_model), include_optimizer=True, overwrite=True)

    
    def hyper_opt(self):
        # setup sherpa object
        if self.ML == 'LSTM':
            parameters = [sherpa.Ordinal(name='neurons', range=[24, 48, 96, 192]),
                        sherpa.Ordinal(name='hidden', range=[1, 2, 3, 4, 5])]
        elif self.ML == 'CNN':
            parameters = [sherpa.Ordinal(name='filters', range=[8, 16, 24]),
                        sherpa.Ordinal(name='neurons', range=[24, 48, 96, 192]),
                        sherpa.Ordinal(name='hidden', range=[1, 2, 3, 4, 5])]
            
        # if self.loss_fn == 'gumbel':
        #     parameters.append(sherpa.Ordinal(name='gamma', range=[0.1, 0.5, 1, 2, 5, 10]))
        # elif self.loss_fn == 'frechet':
        #     parameters.append(sherpa.Ordinal(name='alpha', range=[0.1, 0.5, 1, 2, 5, 10]))
        #     parameters.append(sherpa.Ordinal(name='s', range=[0.1, 0.5, 1, 2, 5, 10]))

        alg = sherpa.algorithms.RandomSearch(max_num_trials=100)
        study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=True, disable_dashboard=True)

        count = 1
        for trial in study:
            self.neurons = trial.parameters['neurons']
            if self.ML == 'CNN' or self.ML == 'ConvLSTM':
                self.filters = trial.parameters['filters']
            self.n_layers = trial.parameters['hidden']
            # drop_value = trial.parameters['dropout']
            # l2 = trial.parameters['l2']
            # batch = trial.parameters['batch']
            self.batch_size = 10 * 24

            self.design_network()
            self.compile()
            # fit network
            self.train_model(hyper_opt=study.keras_callback(trial, objective_name='val_loss'))

            study.finalize(trial)
            if self.logger:
                self.logger.info(f'Trial {self.ML}: {count}')
            else:
                print(f'\nTrial {self.ML}: {count}\n')
            count += 1
        if self.logger:
            self.logger.info(study.get_best_result())
        else:
            print(study.get_best_result())
        study.save(self.sherpa_output)
        # sherpa.Study.load_dashboard(".")
        # ssh -L 8000:localhost:8880 timothyt@cartesius.surfsara.nl
        
    def predict(self, ensemble_loop):
        # Predict for each station
        for station in self.station_inputs.values():
            print(f'\nPredicting station: {station.name}\n')
            station.predict(self.model, ensemble_loop, self.mask_val)


