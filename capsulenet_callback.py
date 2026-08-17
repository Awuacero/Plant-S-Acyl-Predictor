import numpy as np
import tf_keras as keras
from tf_keras import layers, optimizers, callbacks
from tf_keras.models import Model, Sequential
from tf_keras import backend as K
from tf_keras.layers import Dropout, Activation, Add, Layer, BatchNormalization
from tf_keras.callbacks import EarlyStopping
from tf_keras.regularizers import l1, l2, l1_l2

K.set_image_data_format('channels_last')

from capsulelayers import CapsuleLayer, CapsuleLayer_nogradient_stop, PrimaryCap, Length, Mask
from LossCheckPoint import LossModelCheckpoint


class LearningRate(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.learningrate = 0

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.learning_rate)
        self.learningrate = lr


class Extract_outputs(Layer):
    def __init__(self, outputdim=0, **kwargs):
        self.outputdim = outputdim
        super(Extract_outputs, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[1], self.outputdim])

    def call(self, x, mask=None):
        return x[:, :, :self.outputdim]

    def get_config(self):
        config = {'outputdim': self.outputdim}
        base_config = super(Extract_outputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Extract_weight_c(Layer):
    def __init__(self, outputdim, **kwargs):
        self.outputdim = outputdim
        super(Extract_weight_c, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[1], input_shape[-1] - self.outputdim])

    def call(self, x, mask=None):
        return x[:, :, self.outputdim:]

    def get_config(self):
        config = {'outputdim': self.outputdim}
        base_config = super(Extract_weight_c, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def CapsNet(input_shape, n_class, routings, modeltype, power=2):
    if modeltype == "nogradientstop":
        return CapsNet_nogradientstop(input_shape, n_class, routings)
    if modeltype == "nogradientstop_crossentropy":
        return CapsNet_nogradientstop_crossentropy(input_shape, n_class, routings)


def CapsNet_nogradientstop(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal', activation='relu', name='conv1')(x)
    conv1 = Dropout(0.7)(conv1)
    conv2 = layers.Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal', activation='relu', name='conv2')(conv1)
    conv2 = Dropout(0.75)(conv2)
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=60, kernel_size=20, kernel_initializer='he_normal', strides=1, padding='valid', dropout=0.2)
    dim_capsule_dim2 = 10

    digitcaps_c = CapsuleLayer_nogradient_stop(num_capsule=n_class, dim_capsule=dim_capsule_dim2, num_routing=routings, name='digitcaps', kernel_initializer='he_normal', dropout=0.1)(primarycaps)
    digitcaps = Extract_outputs(dim_capsule_dim2)(digitcaps_c)
    out_caps = Length(name='capsnet')(digitcaps)

    return Model(x, out_caps)


def CapsNet_nogradientstop_crossentropy(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal', activation='relu', name='conv1')(x)
    conv1 = Dropout(0.7)(conv1)
    conv2 = layers.Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal', activation='relu', name='conv2')(conv1)
    conv2 = Dropout(0.75)(conv2)
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=60, kernel_size=20, kernel_initializer='he_normal', strides=1, padding='valid', dropout=0.2)
    dim_capsule_dim2 = 10

    digitcaps_c = CapsuleLayer_nogradient_stop(num_capsule=n_class, dim_capsule=dim_capsule_dim2, num_routing=routings, name='digitcaps', kernel_initializer='he_normal', dropout=0.1)(primarycaps)
    digitcaps = Extract_outputs(dim_capsule_dim2)(digitcaps_c)
    out_caps = Length()(digitcaps)
    out_caps = Activation('softmax', name='capsnet')(out_caps)

    return Model(x, out_caps)


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def Capsnet_main(trainX, trainY, valX=None, valY=None, nb_classes=2, nb_epoch=500, earlystop=None, weights=None, compiletimes=0, compilemodels=None, lr=0.001, lrdecay=1, batch_size=500, lam_recon=0.392, routings=3, modeltype='nogradientstop', class_weight=None, activefun='linear', power=2, predict=False, outputweights=None, monitor_file=None, save_best_only=True, load_average_weight=False):
    if len(trainX.shape) > 3:
        trainX.shape = (trainX.shape[0], trainX.shape[2], trainX.shape[3])

    if valX is not None and len(valX.shape) > 3:
        valX.shape = (valX.shape[0], valX.shape[2], valX.shape[3])

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (lrdecay ** epoch))

    if compiletimes == 0:
        model = CapsNet(input_shape=trainX.shape[1:], n_class=nb_classes, routings=routings, modeltype=modeltype)
        if "crossentropy" in str(modeltype):
            model.compile(optimizer=optimizers.Adam(learning_rate=lr, epsilon=1e-08), loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizers.Adam(learning_rate=lr, epsilon=1e-08), loss=margin_loss, metrics=['accuracy'])
    else:
        model = compilemodels

    if not predict:
        if weights is not None and compiletimes == 0:
            model.load_weights(weights)

        weight_checkpointer = LossModelCheckpoint(
            model_file_path=outputweights + '_iteration' + str(compiletimes),
            monitor_file_path=monitor_file + '_iteration' + str(compiletimes) + '.json',
            verbose=1, save_best_only=save_best_only, monitor='val_loss', mode='min', save_weights_only=True
        )

        callbacks_list = [lr_decay, weight_checkpointer]
        if earlystop is not None:
            callbacks_list.append(EarlyStopping(monitor='val_loss', patience=earlystop))
            nb_epoch = 10000

        validation_data = [valX, valY] if valX is not None else None
        model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epoch,
                  validation_data=validation_data, class_weight=class_weight, callbacks=callbacks_list)

        model.load_weights(outputweights + '_iteration' + str(compiletimes))

    return model