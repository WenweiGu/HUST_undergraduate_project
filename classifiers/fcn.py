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


class Classifier_FCN_origin:

    def __init__(self, output_directory, input_shape, nb_classes, y_true):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.y_true = y_true
        self.callbacks_pretrain = None
        self.callbacks = None
        self.nb_classes = nb_classes
        self.model = None
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model_origin.hdf5'

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
        nb_epochs = 2000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        self.model = self.build_model(self.input_shape, self.nb_classes)

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=False, callbacks=self.callbacks)

        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory + 'best_model_origin.hdf5')

        y_pred = model.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, self.y_true, duration, lr=True)

        keras.backend.clear_session()

    def predict(self, x_test):
        model_path = self.output_directory + 'best_model_origin.hdf5'
        model = keras.models.load_model(model_path)
        x_pred = model.predict(x_test)

        return x_pred


class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, y_true, fixed=False):
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

        conv1 = keras.layers.Conv1D(filters=8, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=16, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(filters=8, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        # delete softmax layer
        output_layer = keras.layers.Conv1D(filters=1, kernel_size=5, padding='same')(conv3)

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

    # only use encoder part

    def build_model(self, fixed):
        model_path = self.output_directory + 'best_model_pretrain.hdf5'
        model = keras.models.load_model(model_path)
        model = keras.models.Model(inputs=model.input, outputs=model.layers[2].output)
        input_layer = model.input
        # output_encoder = model.layers[2].output
        # gap_layer = keras.layers.GlobalAveragePooling1D()(output_encoder)
        output_encoder = keras.layers.Flatten()(model.output)

        print(output_encoder.shape)

        layer1 = keras.layers.Dense(64, activation='relu')(output_encoder)

        # layer2 = keras.layers.Dense(32, activation='relu')(layer1)

        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(layer1)

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
        nb_epochs = 2000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        self.model = self.build_model(self.fixed)

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=False, callbacks=self.callbacks)

        duration = time.time() - start_time

        y_pred = self.predict(x_test)

        y_pred = np.argmax(y_pred, axis=1)
        print(y_pred)
        print(self.y_true)

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
