# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import time
from utils.utils import save_logs
import numpy as np
import matplotlib.pyplot as plt
import os


class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, y_true, fixed=True):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.y_true = y_true
        self.callbacks_pretrain = None
        self.callbacks = None
        self.nb_classes = nb_classes
        self.pre_train_model = None
        self.model = None
        self.fixed = fixed
        return

    def build_pretrain_model(self, input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=8, kernel_size=16, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=8, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(filters=8, kernel_size=4, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        # delete softmax layer
        output_layer = keras.layers.Conv1D(filters=1, kernel_size=2, padding='same')(conv3)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.001)

        file_path = self.output_directory + 'best_model_pretrain.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks_pretrain = [reduce_lr, model_checkpoint]

        return model

    def fit_pretrain_model(self, x_train):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 16
        nb_epochs = 1000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        self.pre_train_model = self.build_pretrain_model(self.input_shape)

        self.pre_train_model.fit(x_train, x_train, batch_size=mini_batch_size, epochs=nb_epochs,
                                 callbacks=self.callbacks_pretrain, verbose=False)

        x_reconstruct = self.reconstruct(x_train)

        for i in range(x_train.shape[0]):
            plt.close()
            plt.plot(x_train[i])
            plt.plot(x_reconstruct[i])
            if not os.path.exists('./UCR/results/fcn/images/'):
                os.mkdir('./UCR/results/fcn/images/')
            plt.savefig('./UCR/results/fcn/images/' + str(i) + '.png')

        keras.backend.clear_session()

    def build_model(self, fixed):
        model_path = self.output_directory + 'best_model_pretrain.hdf5'
        model = keras.models.load_model(model_path)
        input_layer = model.input
        output_da = model.output
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_da)
        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        if fixed:
            for layer in model.layers:
                layer.trainable = False

            model.layers[-1].trainable = True

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.00001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit_model(self, x_train, y_train, x_test):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 16
        nb_epochs = 3000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        self.model = self.build_model(self.fixed)

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=False, callbacks=self.callbacks)

        duration = time.time() - start_time

        y_pred = self.predict(x_test)

        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, self.y_true, duration, lr=True)

        keras.backend.clear_session()

    def reconstruct(self, x_test):
        model_path = self.output_directory + 'best_model_pretrain.hdf5'
        model = keras.models.load_model(model_path)
        x_reconstruct = model.predict(x_test)

        return x_reconstruct

    def predict(self, x_test):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        x_pred = model.predict(x_test)

        return x_pred
