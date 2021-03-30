import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import CLASSIFIERS
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.interpolate import interp1d

matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def read_ucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        os.makedirs(directory_path)
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def read_dataset(root_dir, dataset_name):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    root_dir_dataset = cur_root_dir + '/UCRArchive_2018/' + dataset_name + '/'
    df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

    df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]

    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])

    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])

    x_train = x_train.values
    x_test = x_test.values

    # z-norm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())

    return datasets_dict


def transform_to_same_length(x, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        ts = x[i]
        curr_length = ts.shape[0]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, max(idx), max_length)
        # linear interpolation
        f = interp1d(idx, ts, axis=0, kind='linear')
        new_ts = f(idx_new)
        ucr_x[i] = new_ts

    return ucr_x


def calculate_metrics(y_pred, y_true, duration):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_pred, y_true, average='macro')
    res['accuracy'] = accuracy_score(y_pred, y_true)

    res['recall'] = recall_score(y_pred, y_true, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_pred, y_true, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_train_acc',
                                          'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    if lr:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics
