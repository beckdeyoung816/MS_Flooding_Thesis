# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:57:17 2020

@author: acn980
"""

#https://keras.io/examples/vision/conv_lstm/


input_shape = (1, lat_,lon_,1)
cnn_input = keras.Input(shape=input_shape)
cnn_lay = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True)(cnn_input)
cnn_lay = layers.BatchNormalization()(cnn_lay)
cnn_lay = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True)(cnn_lay)
cnn_lay = layers.BatchNormalization()(cnn_lay)
cnn_lay = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True)(cnn_lay)
cnn_lay = layers.BatchNormalization()(cnn_lay)
cnn_lay = layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True)(cnn_lay)
cnn_lay = layers.BatchNormalization()(cnn_lay)
#cnn_input = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(cnn_input)
cnn_lay = layers.Flatten()(cnn_lay)
#x = layers.Dense(50, activation='relu')(cnn_lay)
outputs = layers.Dense(1, activation='relu')(cnn_lay)

model = keras.Model(inputs=[cnn_input], outputs=outputs, name=name_model)
model.compile(optimizer='adam', loss='mae')
model.summary()

batch_size=1000#30*24 #72 #We select one month
epochs=20


# fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)


