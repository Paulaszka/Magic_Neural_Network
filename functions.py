import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import tensorflow as tf
from keras.src.utils import to_categorical
from sklearn.metrics import *


def prepare_type_list(y_pred):
    max_indices = []
    for row in y_pred:
        max_index = row.argmax()
        max_indices.append(max_index+1)
    return max_indices


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


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid()
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_global_error(y_test, y_pred_bin):
    counter = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred_bin[i]:
            counter += 1

    result = round(counter / len(y_test), 8)
    return result


def calculate_individual_error(y_test, y_pred_bin, types_list):
    individual_error_list = [0] * len(types_list)
    individual_correct_list = [0] * len(types_list)
    num_of_individuals = [0] * len(types_list)
    correct = 0

    for i in range(len(y_test)):
        for j in range(len(types_list)):
            if y_test[i] != y_pred_bin[i] and y_test[i] == types_list[j]:
                individual_error_list[j] += 1
            elif y_test[i] == y_pred_bin[i] and y_test[i] == types_list[j]:
                individual_correct_list[j] += 1

    for i in range(len(y_test)):
        for j in range(len(types_list)):
            if y_test[i] == types_list[j]:
                num_of_individuals[j] += 1

    for i in range(len(individual_correct_list)):
        correct += individual_correct_list[i]

    # print(types_list, "jakie rodzaje")
    # print(num_of_individuals, "ile elementow kazdego rodzaju")
    # print(individual_error_list, "ile bledow kazdego rodzaju")

    for i in range(len(individual_error_list)):
        individual_error_list[i] = individual_error_list[i] / num_of_individuals[i]

    # print(individual_error_list, "procent bledow kazdego rodzaju")

    return individual_error_list, individual_correct_list, correct


def test_logs(global_error, indiv_error, report, conf_matrix, filename):
    with open(filename, 'w') as plik:
        plik.write("WIELKOSCI Z CZESCI TESTOWEJ\n\nRaport\n")
        plik.write(report)

        plik.write("\nMacierz pomylek\n")
        plik.write(str(conf_matrix))

        plik.write("\n\nBlad dla calego wzorca\n")
        plik.write(str(global_error))

        plik.write("\n\nLista bledow na poszczegolnych wyjsciach \n")
        for element in indiv_error:
            plik.write(f"{element}\n")



def result_logs(y_test, y_pred, indiv_correct, correct, filename):
    with open(filename, 'w') as plik:
        plik.write("POROWNANIE WYNIKOW\nLiczba poprawnie sklasyfikowanych elementow:\n")
        plik.write(str(correct))
        plik.write("\nZ podzialem na klasy:\n")
        for element in indiv_correct:
            plik.write(f"{element}\n")
        plik.write("Wyniki testowe - Wyniki przewidywane\n")
        for test, pred in zip(y_test, y_pred):
            plik.write(f"{test} - {pred}\n")

