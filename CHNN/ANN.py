# -*- coding: utf-8 -*-
"""
Training Neural networks and predicting surge levels

Timothy Tiggeloven and Anaïs Couasnon

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
import numpy as np
import random
import keras.backend as K

def reset_seeds():
    #Solution to reset random states from: https://stackoverflow.com/questions/58453793/the-clear-session-method-of-keras-backend-does-not-clean-up-the-fitting-data 
    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    print("RANDOM SEEDS RESET")

#print('do you see this change?')
def design_network(n_layers, neurons, filters, train_X, dropout, drop_value, variables, batch_normalization, name_model, ML='LSTM', loss='mae', optimizer='adam',
                   activation='relu', summary=False, figures_dir='Figures', model_dir='Models', l1=0.01, l2=0.01, mask_val=-999):
    if activation == 'swish':
        utils.generic_utils.get_custom_objects().update({'swish': swish(swish_func)})

    if activation == 'relu':
        activation = 'relu_advanced'
        utils.generic_utils.get_custom_objects().update({'relu_advanced': relu_advanced(relu_advanced_func)})

    # design network
    if ML == 'LSTM':
        model = LSTM_model(n_layers, neurons, activation, train_X, dropout, drop_value, l1=l1, l2=l2, mask_val=mask_val)
    elif ML == 'CNN':
        model = CNN_model(n_layers, neurons, filters, activation, train_X, variables, name_model, dropout, drop_value, l1=l1, l2=l2)
    elif ML == 'ConvLSTM':
        model = ConvLSTM_model(n_layers, neurons, filters, activation, train_X, variables, batch_normalization, name_model, dropout, drop_value, l1=l1, l2=l2, mask_val=mask_val)
    elif ML == 'ANN':
        model = ANN_model(n_layers, neurons, activation, train_X, variables, name_model, dropout, drop_value, l1=l1, l2=l2)

    model.compile(loss=loss, optimizer=optimizer)
    if summary:
        model.summary()

#    utils.plot_model(model, os.path.join(figures_dir, f'{ML}_layers{n_layers}.png'), show_shapes=True)
    return model


def LSTM_model(n_layers, neurons, activation, train_X, dropout, drop_value, l1=0.01, l2=0.01,
               mask_val=-999):
    # design LSTM
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = models.Sequential()
    # model.add(layers.Masking(mask_value=mask_val, input_shape=input_shape))
    for i in range(n_layers):
        rs = False if i == n_layers - 1 else True
        if activation == 'Leaky ReLu':
            model.add(layers.LSTM(neurons, input_shape=input_shape, return_sequences=rs, activation=activation,  # added this activation to be sure, but not yet tested
                                  stateful=False, recurrent_activation='hard_sigmoid'))  # neurons refers to cells
            model.add(layers.LeakyReLU(alpha=0.1))
            # if dropout:
            #     model.add(layers.Dropout(drop_value))
        else:
            model.add(layers.LSTM(neurons, input_shape=input_shape, activation=activation, 
                      return_sequences=rs, recurrent_activation='hard_sigmoid'))  # recurrent_dropout=drop_value
            # if dropout:
            #     model.add(layers.Dropout(drop_value))
    model.add(layers.Dense(neurons, activation=activation, kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)))  # hidden layers
    if dropout:
        model.add(layers.Dropout(drop_value))
    model.add(layers.Dense(1, activation=activation)) #output layer

    return model


def ConvLSTM_model(n_layers, neurons, filters, activation, train_X, variables, batch_normalization,
                   name_model, dropout, drop_value, l1=0.01, l2=0.01, mask_val=-999):
    input_shape = (1, train_X[0].shape[2], train_X[0].shape[3], 1)

    merge_list, input_list = [], []
    for i, var in enumerate(variables):
        cnn_input = keras.Input(shape=input_shape)
        cnn_mask = layers.Masking(mask_value=mask_val).compute_mask(cnn_input)

        #First layer
        cnn_lay = layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding="same",
                            return_sequences=True, activation=activation)(cnn_input, mask=cnn_mask)  # https://stackoverflow.com/questions/43392693/how-to-input-mask-value-to-convolution1d-layer
        if batch_normalization:
            cnn_lay = layers.BatchNormalization()(cnn_lay)
        cnn_lay= layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")(cnn_lay)

        #Other layers
        for j in range(n_layers-1): 
            cnn_lay = layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding="same",
                                        return_sequences=True, activation=activation)(cnn_lay) # Masking should be propagated
            if batch_normalization:
                cnn_lay = layers.BatchNormalization()(cnn_lay)
            cnn_lay= layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")(cnn_lay)

        cnn_lay = layers.Flatten()(cnn_lay)
        # if dropout:
        #     cnn_lay= layers.Dropout(drop_value)(cnn_lay)
        if len(variables) > 1:
            merge_list.append(cnn_lay)
            input_list.append(cnn_input)
        else:
            merge_list = cnn_lay
            input_list = cnn_input

    if len(variables) > 1:
        merge_list = layers.concatenate(merge_list)
    hidden = layers.Dense(neurons, activation=activation, kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2))(merge_list) # hidden layer
    if dropout:
        hidden = layers.Dropout(drop_value)(hidden)
    outputs = layers.Dense(1)(hidden)
    model = keras.Model(inputs=input_list, outputs=outputs, name=name_model)
    return model


def CNN_model(n_layers, neurons, filters, activation, train_X, variables, name_model, dropout,
              drop_value, l1=0.01, l2=0.01):
    input_shape = (train_X[0].shape[1], train_X[0].shape[2], 1)

    merge_list, input_list = [], []
    for i, var in enumerate(variables):
        cnn_input = keras.Input(shape=input_shape)
        filters_l = filters
        cnn1= layers.Conv2D(filters_l, kernel_size=(3, 3), padding='same', activation=activation)(cnn_input)     
        cnn1= layers.MaxPooling2D(pool_size=(2, 2), padding="same")(cnn1)
        for i in range(n_layers - 1):
            # filters_l = filters_l * 2
            cnn1= layers.Conv2D(filters_l, kernel_size=(3, 3), padding='same', activation=activation)(cnn1)
            cnn1= layers.MaxPooling2D(pool_size=(2, 2), padding="same")(cnn1)
        cnn1 = layers.Flatten()(cnn1)
        # if dropout:
        #     cnn1= layers.Dropout(drop_value)(cnn1)
        if len(variables) > 1:
            merge_list.append(cnn1)
            input_list.append(cnn_input)
        else:
            merge_list = cnn1
            input_list = cnn_input

    if len(variables) > 1:
        merge_list = layers.concatenate(merge_list)
    x = layers.Dense(neurons, kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2), activation=activation)(merge_list)  # hidden layer
    if dropout:
        x = layers.Dropout(drop_value)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=input_list, outputs=outputs, name=name_model)

    return model


def ANN_model(n_layers, neurons, activation, train_X, variables, name_model, dropout, drop_value,
              l1=0.01, l2=0.01):
    input_shape = (train_X.shape[1],)

    ann_input = keras.Input(shape=input_shape)
    # if hasattr(neurons, "__getitem__"): #Multiple hidden layers
    #     x = layers.Dense(neurons[0], kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2), activation=activation)(ann_input)  # hidden layer
    #     for i in range(n_layers - 1):
    #         x = layers.Dense(neurons[i+1], kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2), activation=activation)(x)
    # else:
    x = layers.Dense(neurons, kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2), activation=activation)(ann_input)
    for i in range(n_layers - 1):
        x = layers.Dense(neurons, kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2), activation=activation)(x)
        
    if dropout:
        x = layers.Dropout(drop_value)(x)
        
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=ann_input, outputs=outputs, name=name_model)

    return model


def train_model(model, epochs, batch, train_X, train_y, test_X, test_y, ML, name_model, model_dir,
                validation='split', verbose=2, hyper_opt=False):
    # reset_seeds()
    
    #'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    if os.path.exists(os.path.join(model_dir, name_model)):
        os.remove(os.path.join(model_dir, name_model))
        
    my_callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode="auto", restore_best_weights = 'True'),
                    ModelCheckpoint(filepath=os.path.join(model_dir, name_model), monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)] # ProgbarLogger(count_mode="steps", stateful_metrics=None), , ModelCheckpoint(filepath=os.path.join(model_dir, name_model), monitor='loss', save_best_only=True, save_weights_only=False, mode='auto', period=1),
    # my_callbacks = [ModelCheckpoint(filepath=os.path.join(model_dir, name_model), monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)]
    
    if hyper_opt:
        my_callbacks.append(hyper_opt)

    if ML == 'LSTM' or ML == 'ConvLSTM': 
        shuffle = 'batch'
    else:
        shuffle= True
    
    # fit network
    if validation == 'split':
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, workers=3, use_multiprocessing=True, # validation_data=(test_X, test_y)
                            validation_split=0.3, callbacks=my_callbacks, verbose=verbose, shuffle=shuffle) # , workers=3, use_multiprocessing=True,
    else:
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, workers=3, use_multiprocessing=True,
                            validation_data=(test_X, test_y), callbacks=my_callbacks, verbose=verbose, shuffle=shuffle)

    # model.save(os.path.join(model_dir, name_model), include_optimizer=True) # , overwrite=True

    # loss values
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    del(history)

    return model, train_loss, test_loss


def predict(model, test_X, reframed_df, scaler, n_train_final):
    df = reframed_df[n_train_final:].copy()

    # make a prediction
    yhat = model.predict(test_X)

    # invert scaling for observed surge
    inv_y = scaler.inverse_transform(df.values)[:,-1]

    # invert scaling for modelled surge
    df.loc[:,'values(t)'] = yhat
    inv_yhat = scaler.inverse_transform(df.values)[:,-1]

    return inv_yhat, inv_y


def hyper_opt(train_X, train_y, val_X, val_y, l1, l2, activation, loss, optimizer, neurons,
              drop_value, epochs, batch, verbose, model_dir, name_model, ML, filters, dropout,
              n_layers, variables, batch_normalization, sherpa_output, logger):
    # setup sherpa object
    if ML == 'LSTM' or ML == 'ANN':
        parameters = [sherpa.Ordinal(name='neurons', range=[24, 48, 96, 192]),
                      sherpa.Ordinal(name='hidden', range=[1, 2, 3, 4, 5])]
    elif ML == 'CNN' or ML == 'ConvLSTM':
        parameters = [sherpa.Ordinal(name='filters', range=[8, 16, 24]),
                      sherpa.Ordinal(name='neurons', range=[24, 48, 96, 192]),
                      sherpa.Ordinal(name='hidden', range=[1, 2, 3, 4, 5])]
    # parameters = [sherpa.Ordinal(name='dropout', range=[0., 0.1, 0.2, 0.5]),
    #               sherpa.Ordinal(name='neurons', range=[24, 48, 96]),
    #               sherpa.Ordinal(name='batch', range=[10 * 24, 50 * 24, 100 * 24]),
    #               sherpa.Ordinal(name='l2', range=[0, 0.001, 0.01, 0.1])]
    # parameters = [sherpa.Ordinal(name='filters', range=[8, 16, 24]),
    #               sherpa.Ordinal(name='neurons', range=[24, 48, 96])]
    # parameters = [sherpa.Ordinal(name='l2', range=[0.001, 0.01, 0.1])]
    alg = sherpa.algorithms.RandomSearch(max_num_trials=100)
    study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=True, disable_dashboard=True)

    count = 1
    for trial in study:
        neurons = trial.parameters['neurons']
        if ML == 'CNN' or ML == 'ConvLSTM':
            filters = trial.parameters['filters']
        n_layers = trial.parameters['hidden']
        # drop_value = trial.parameters['dropout']
        # l2 = trial.parameters['l2']
        # batch = trial.parameters['batch']
        batch = 10 * 24

        model = design_network(n_layers, neurons, filters, train_X, dropout,
                               drop_value, variables,
                               batch_normalization, name_model, ML=ML, loss=loss,
                               optimizer=optimizer, activation=activation, l1=l1, l2=l2)
        
        # fit network
        model, train_loss, test_loss = train_model(model, epochs, batch, train_X, train_y, val_X,
                                                   val_y, ML, name_model, model_dir,
                                                   validation='select', verbose=verbose,
                                                   hyper_opt=study.keras_callback(trial, objective_name='val_loss'))

        study.finalize(trial)
        if logger:
            logger.info(f'Trial {ML}: {count}')
        else:
            print(f'\nTrial {ML}: {count}\n')
        count += 1
    if logger:
        logger.info(study.get_best_result())
    else:
        print(study.get_best_result())
    study.save(sherpa_output)
    # sherpa.Study.load_dashboard(".")
    # ssh -L 8000:localhost:8880 timothyt@cartesius.surfsara.nl


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


