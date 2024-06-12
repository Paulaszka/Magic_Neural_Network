import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def prepare_data(file_path):
    cols = ["sepal length", "sepal width", "petal length", "petal width", "iris type"]
    data_file = pd.read_csv(file_path, names=cols, header=None)
    for wiersz in data_file:
        ostatnia_wartosc = wiersz[-1]
        print(ostatnia_wartosc)
    mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    data_file['iris type'] = data_file['iris type'].replace(mapping).infer_objects(copy=False)
    return data_file


def oversample_set(data_set, oversample=False):
    x = data_set[data_set.columns[:-1]].values
    y = data_set[data_set.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    connected = np.hstack((x, np.reshape(y, (-1, 1))))
    return connected, x, y



