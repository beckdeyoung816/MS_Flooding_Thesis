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
import os

def design_network(n_layers, neurons, train_X, dropout, drop_value, variables, ML='LSTM', loss='mae', optimizer='adam',
				   activation='relu', summary=False, figures_dir='Figures', model_dir='Models'):
    print(f'Building model: {ML}\n')
    if activation == 'swish':
        utils.generic_utils.get_custom_objects().update({'swish': swish(swish_func)})

    # design network
    if ML == 'LSTM':
		model = LSTM_model(n_layers, neurons, activation, train_X, dropout, drop_value)
    elif ML == 'CNN':
		model = CNN_model(n_layers, neurons, activation, train_X)
    elif ML == 'ConvLSTM':
		model = ConvLSTM_model(n_layers, neurons, activation, train_X, variables, ML)
	
    model.compile(loss=loss, optimizer=optimizer)
    if summary:
	    model.summary()

    utils.plot_model(model, os.path.join(figures_dir, f'{ML}.png'), show_shapes=True)

    return model

def LSTM_model(n_layers, neurons, activation, train_X, dropout, drop_value):
	# design LSTM
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = models.Sequential()
    if n_layers > 1:
        for i in range(n_layers - 1):
            if activation == 'Leaky ReLu':
                model.add(layers.LSTM(neurons, input_shape=input_shape, return_sequences=True))
                model.add(layers.LeakyReLU(alpha=0.1))
                if dropout:
                    model.add(layers.Dropout(drop_value))
            else:
                model.add(layers.LSTM(neurons, input_shape=input_shape, activation=activation,
                                      return_sequences=True))
                if dropout:
                    model.add(layers.Dropout(drop_value))

    if activation == 'Leaky ReLu':
        model.add(layers.LSTM(neurons, input_shape=input_shape))
        model.add(layers.LeakyReLU(alpha=0.1))
        if dropout:
            model.add(layers.Dropout(drop_value))
    else:
        model.add(layers.LSTM(neurons, input_shape=input_shape, activation=activation))
        if dropout:
            model.add(layers.Dropout(drop_value))
	model.add(layers.Dense(1))

	return model

def ConvLSTM_model(n_layers, neurons, activation, train_X, variables, ML):
    input_shape = (1, train_X[0].shape[2], train_X[0].shape[3], 1)

    merge_list, input_list = [], []
    for i, var in enumerate(variables):
        cnn_input = keras.Input(shape=input_shape)
        for i in range(n_layers - 1):
            cnn_lay = layers.ConvLSTM2D(filters=neurons, kernel_size=(3, 3), padding="same",
                                        return_sequences=True, activation=activation,
                                        recurrent_activation=activation)(cnn_input)
            # cnn_lay = layers.BatchNormalization()(cnn_lay)
            cnn_lay = layers.ConvLSTM2D(filters=neurons, kernel_size=(3, 3), padding="same",
                                        return_sequences=True, activation=activation,
                                        recurrent_activation=activation)(cnn_input)
            # cnn_lay = layers.BatchNormalization()(cnn_lay)
        cnn_lay = layers.Flatten()(cnn_lay)
        merge_list.append(cnn_lay)
        input_list.append(cnn_input)

	merge = layers.concatenate(merge_list)
	outputs = layers.Dense(1)(merge)
	model = keras.Model(inputs=input_list, outputs=outputs, name=ML)

	return model

def CNN_model(n_layers, neurons, activation, train_X):
	input_shape = (train_X[0].shape[1], train_X[0].shape[2], 1)

	merge_list, input_list = [], []
	for i, var in enumerate(variables):
		cnn_input = keras.Input(shape=input_shape)
		cnn1= layers.Conv2D(neurons, kernel_size=(3, 3), padding='same', activation=activation)(cnn_input)
		cnn1= layers.MaxPooling2D(pool_size=(2, 2))(cnn1)
		cnn1= layers.Conv2D(neurons*2, kernel_size=(3, 3), padding='same', activation=activation)(cnn1)
		cnn1= layers.MaxPooling2D(pool_size=(2, 2))(cnn1)
		cnn1= layers.Flatten()(cnn1)
		if dropout:
			cnn1= layers.Dropout(drop_value)(cnn1)
		merge_list.append(cnn1)
		input_list.append(cnn_input)

	merge = layers.concatenate(merge_list)
	x = layers.Dense(50, kernel_regularizer=regularizers.l2(0.01), activation=activation)(merge)
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(1)(x)
	model = keras.Model(inputs=input_list, outputs=outputs, name=name_model)

	return model

def train_data(model, epochs, batch, train_X, train_y, test_X, test_y):
	# fit network
	if ML == 'LSTM' or ML == 'ConvLSTM':
		history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, use_multiprocessing=True,
							validation_data=(test_X, test_y), verbose=2, shuffle=False, workers=3)
	elif ML == 'CNN':
		history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, shuffle=True,
							validation_split=0.3, workers=3, verbose=2, use_multiprocessing=True)
	
	model.save(os.path.join(workspace, path_model, name_model), overwrite=True, include_optimizer=True)

	# loss values
	train_loss = history.history['loss']
	test_loss = history.history['val_loss']

	return model, train_loss, test_loss

def forescast(model, test_X, reframed_df):
	# make a prediction
	yhat = model.predict(test_X)

	# invert scaling for actual
	inv_y = scaler.inverse_transform(reframed_df.values)[:,-1]

	# invert scaling for actual
	reframed_df['values(t)'] = yhat
	inv_yhat = scaler.inverse_transform(reframed_df.values)[:,-1]

	# # invert scaling for forecast
	# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
	# inv_yhat = np.concatenate((test_X[:, :], yhat), axis=1)
	# inv_yhat = scaler.inverse_transform(inv_yhat)
	# inv_yhat = inv_yhat[:,-1]
	# a = scaler.inverse_transform(reframed_df.values)

	# # invert scaling for actual
	# inv_y = test_y.reshape((len(test_y), 1))
	# inv_y = np.concatenate((test_X[:, :], inv_y), axis=1)
	# inv_y = scaler.inverse_transform(inv_y)
	# inv_y = inv_y[:,-1]

	return inv_yhat, inv_y


def swish_func(x, beta = 1):
    return (x * sigmoid(beta * x))

class swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

