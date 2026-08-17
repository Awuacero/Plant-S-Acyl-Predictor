import os
import time
import numpy as np
import pandas as pd

import tf_keras as keras
from tf_keras import layers
from tf_keras.models import Model
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tf_keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv1D, Reshape, BatchNormalization, concatenate
from tf_keras.optimizers import Adam, SGD
from tf_keras.regularizers import l1, l2

from attention import Attention, myFlatten
from LossCheckPoint import LossModelCheckpoint


def copy_model(input_row, input_col):
    filtersize1 = 1
    filtersize2 = 9
    filtersize3 = 10
    filter1 = 200
    filter2 = 150
    filter3 = 200
    dropout1 = 0.75
    dropout2 = 0.75
    dropout4 = 0.75
    dropout5 = 0.75
    dropout6 = 0
    L1CNN = 0
    nb_classes = 2
    actfun = "relu"
    attentionhidden_x = 10
    attentionhidden_xr = 8
    attention_reg_x = 0.151948
    attention_reg_xr = 2
    dense_size1 = 149
    dense_size2 = 8
    dropout_dense1 = 0.298224
    dropout_dense2 = 0

    inp = Input(shape=(input_row, input_col))
    x = Conv1D(filter1, filtersize1, kernel_initializer='he_normal', kernel_regularizer=l1(L1CNN), padding="same")(inp)
    x = Dropout(dropout1)(x)
    x = Activation(actfun)(x)
    
    x = Conv1D(filter2, filtersize2, kernel_initializer='he_normal', kernel_regularizer=l1(L1CNN), padding="same")(x)
    x = Dropout(dropout2)(x)
    x = Activation(actfun)(x)
    
    x = Conv1D(filter3, filtersize3, kernel_initializer='he_normal', kernel_regularizer=l1(L1CNN), padding="same")(x)
    x = Activation(actfun)(x)

    x_dim1, x_dim2 = int(x.shape[1]), int(x.shape[2])
    x_reshape = Reshape((x_dim2, x_dim1))(x)
    x = Dropout(dropout4)(x)
    x_reshape = Dropout(dropout5)(x_reshape)

    decoder_x = Attention(hidden=attentionhidden_x, activation='linear', init='he_normal', W_regularizer=l1(attention_reg_x))
    decoded_x = decoder_x(x)
    output_x = myFlatten(x_dim2)(decoded_x)

    decoder_xr = Attention(hidden=attentionhidden_xr, activation='linear', init='he_normal', W_regularizer=l1(attention_reg_xr))
    decoded_xr = decoder_xr(x_reshape)
    output_xr = myFlatten(x_dim1)(decoded_xr)

    output = concatenate([output_x, output_xr])
    output = Dropout(dropout6)(output)
    output = Dense(dense_size1, kernel_initializer='he_normal', activation='relu')(output)
    output = Dropout(dropout_dense1)(output)
    output = Dense(dense_size2, activation="relu", kernel_initializer='he_normal')(output)
    output = Dropout(dropout_dense2)(output)
    out = Dense(nb_classes, kernel_initializer='he_normal', activation='softmax')(output)

    return Model(inp, out)


def MultiCNN(trainX, trainY, valX=None, valY=None,
             nb_classes=2, nb_epoch=500, earlystop=None,
             weights=None, compiletimes=0, compilemodels=None,
             batch_size=1000,
             class_weight=None,
             transferlayer=1, forkinase=False,
             predict=False,
             outputweights=None,
             monitor_file=None,
             save_best_only=True,
             load_average_weight=False):

    if len(trainX.shape) > 3:
        trainX.shape = (trainX.shape[0], trainX.shape[2], trainX.shape[3])

    if valX is not None and len(valX.shape) > 3:
        valX.shape = (valX.shape[0], valX.shape[2], valX.shape[3])

    if compiletimes == 0:
        filtersize1 = 1
        filtersize2 = 9
        filtersize3 = 10
        filter1 = 200
        filter2 = 150
        filter3 = 200
        dropout1 = 0.75
        dropout2 = 0.75
        dropout4 = 0.75
        dropout5 = 0.75
        dropout6 = 0
        L1CNN = 0
        actfun = "relu"
        attentionhidden_x = 10
        attentionhidden_xr = 8
        attention_reg_x = 0.151948
        attention_reg_xr = 2
        dense_size1 = 149
        dense_size2 = 8
        dropout_dense1 = 0.298224
        dropout_dense2 = 0

        inp = Input(shape=(trainX.shape[1], trainX.shape[2]))
        x = Conv1D(filter1, filtersize1, kernel_initializer='he_normal', kernel_regularizer=l1(L1CNN), padding="same")(inp)
        x = Dropout(dropout1)(x)
        x = Activation(actfun)(x)

        x = Conv1D(filter2, filtersize2, kernel_initializer='he_normal', kernel_regularizer=l1(L1CNN), padding="same")(x)
        x = Dropout(dropout2)(x)
        x = Activation(actfun)(x)

        x = Conv1D(filter3, filtersize3, kernel_initializer='he_normal', kernel_regularizer=l1(L1CNN), padding="same")(x)
        x = Activation(actfun)(x)

        x_dim1, x_dim2 = int(x.shape[1]), int(x.shape[2])
        x_reshape = Reshape((x_dim2, x_dim1))(x)

        x = Dropout(dropout4)(x)
        x_reshape = Dropout(dropout5)(x_reshape)

        decoder_x = Attention(hidden=attentionhidden_x, activation='linear', init='he_normal', W_regularizer=l1(attention_reg_x))
        decoded_x = decoder_x(x)
        output_x = myFlatten(x_dim2)(decoded_x)

        decoder_xr = Attention(hidden=attentionhidden_xr, activation='linear', init='he_normal', W_regularizer=l1(attention_reg_xr))
        decoded_xr = decoder_xr(x_reshape)
        output_xr = myFlatten(x_dim1)(decoded_xr)

        output = concatenate([output_x, output_xr])
        output = Dropout(dropout6)(output)
        output = Dense(dense_size1, kernel_initializer='he_normal', activation='relu')(output)
        output = Dropout(dropout_dense1)(output)
        output = Dense(dense_size2, activation="relu", kernel_initializer='he_normal')(output)
        output = Dropout(dropout_dense2)(output)
        out = Dense(nb_classes, kernel_initializer='he_normal', activation='softmax')(output)

        cnn = Model(inp, out)
        cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        cnn = compilemodels

    if not predict:
        if weights is not None and compiletimes == 0:
            if not forkinase:
                cnn.load_weights(weights)
            else:
                cnn2 = copy_model(trainX.shape[1], trainX.shape[2])
                cnn2.load_weights(weights)
                for l in range(len(cnn2.layers) - transferlayer):
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())

        weight_checkpointer = LossModelCheckpoint(
            model_file_path=outputweights + '_iteration' + str(compiletimes),
            monitor_file_path=monitor_file + '_iteration' + str(compiletimes) + '.json',
            verbose=1, save_best_only=save_best_only,
            monitor='val_loss', mode='min',
            save_weights_only=True
        )

        callbacks_list = [weight_checkpointer]
        if earlystop is not None:
            callbacks_list.append(EarlyStopping(monitor='val_loss', patience=earlystop))
            nb_epoch = 10000

        validation_data = (valX, valY) if valX is not None else None
        cnn.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epoch,
                validation_data=validation_data, callbacks=callbacks_list)

        cnn.load_weights(outputweights + '_iteration' + str(compiletimes))

    return cnn