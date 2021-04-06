from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_to_same_length
import numpy as np
import pandas as pd
import sklearn
import random
import os


def fit_classifier(dataset, dataset_list=None):
    dataset_dict = read_dataset(root_dir, dataset)
    x_train_origin = dataset_dict[dataset][0]
    y_train_origin = dataset_dict[dataset][1]
    x_test_origin = dataset_dict[dataset][2]
    y_test_origin = dataset_dict[dataset][3]

    x_train_origin = x_train_origin.reshape((x_train_origin.shape[0], x_train_origin.shape[1], 1))
    x_test_origin = x_test_origin.reshape((x_test_origin.shape[0], x_test_origin.shape[1], 1))

    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train_origin, y_test_origin), axis=0).reshape(-1, 1))
    y_train_origin = enc.transform(y_train_origin.reshape(-1, 1)).toarray()

    x_train = dataset_dict[dataset][0]
    y_train = dataset_dict[dataset][1]
    x_test = dataset_dict[dataset][2]
    y_test = dataset_dict[dataset][3]

    timestamp = []

    for data_pretrain in dataset_list:
        data = read_dataset(root_dir, data_pretrain)
        length = data[data_pretrain][0].shape[1]
        timestamp.append(length)

    length_max = max(timestamp)

    x_train = transform_to_same_length(x_train, length_max)

    x_test = transform_to_same_length(x_test, length_max)

    x_pretrain = None

    if dataset_list:
        dataset_list.remove(dataset)

        x_pretrain = transform_to_same_length(np.copy(x_train), length_max)

        for data_pretrain in dataset_list:
            read_data = read_dataset(root_dir, data_pretrain)
            data = read_data[data_pretrain][0]
            data = transform_to_same_length(data, length_max)
            x_pretrain = np.concatenate((x_pretrain, data), axis=0)

        x_pretrain = x_pretrain.reshape((x_pretrain.shape[0], x_pretrain.shape[1], 1))

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    y_true = np.argmax(y_test, axis=1)

    input_shape_origin = x_train_origin.shape[1:]
    input_shape = x_train.shape[1:]
    classifier_origin = create_classifier(classifier_name, input_shape_origin, nb_classes, output_directory, y_true,
                                          False)[0]
    classifier_origin.fit_model(x_train_origin, y_train_origin, x_test_origin)

    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, y_true, False)[1]
    classifier.fit_pretrain_model(x_pretrain)
    classifier.fit_model(x_train, y_train, x_test)


def create_classifier(classifier, input_shape, nb_classes, output_dir, y_test, verbose=False):
    if classifier == 'fcn':
        from classifiers import fcn
        model_origin = fcn.Classifier_FCN_origin(output_dir, input_shape, nb_classes, y_test)
        model = fcn.Classifier_FCN(output_dir, input_shape, nb_classes, y_test)
        return model_origin, model
    if classifier == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_dir, input_shape, nb_classes, verbose)
    if classifier == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_dir, input_shape, nb_classes, verbose)
    if classifier == 'cnn':
        from classifiers import cnn
        return cnn.Classifier_CNN(output_dir, input_shape, nb_classes, verbose)


def data_augmentation(augment_sample_number, x_train, y_train):
    N = x_train.shape[0]
    length_max = x_train.shape[1]
    idx_to_augment = random.sample(range(0, N), augment_sample_number)
    segment_len = length_max // 2
    x_train_augmented = np.zeros((augment_sample_number, segment_len))

    for i in range(len(idx_to_augment)):
        start_point = random.sample(range(0, length_max - segment_len + 1), 1)[0]
        temp = x_train[idx_to_augment[i]][start_point:start_point + segment_len].reshape((1, segment_len))
        x_train_augmented[i] = temp

    x_train_augmented = x_train_augmented.reshape((x_train_augmented.shape[0], x_train_augmented.shape[1], 1))
    x_train_augmented = transform_to_same_length(x_train_augmented, length_max)
    y_train_augmented = y_train[idx_to_augment]

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    x_train = np.concatenate((x_train, x_train_augmented), axis=0)
    y_train = np.concatenate((y_train, y_train_augmented), axis=0)
    return x_train, y_train


# change this directory for your machine
root_dir = './UCR'

# this is the code used to launch an experiment on a dataset
dataset_name = 'Coffee'
classifier_name = 'fcn'
data_name_list = ['ACSF1', 'Adiac',
                  'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                  'Herring']

output_directory = root_dir + '/results/' + classifier_name + '/' + \
                   dataset_name + '/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

if os.path.exists(test_dir_df_metrics):
    csv = pd.read_csv(test_dir_df_metrics)
    os.remove(test_dir_df_metrics)

print('Method: ', dataset_name, classifier_name)

create_directory(output_directory)

# feed single dataset
fit_classifier(dataset_name, data_name_list)

print('DONE')

# the creation of this directory means
create_directory(output_directory + '/DONE')
