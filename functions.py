import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# - - - DATA PREPARATION - - -

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


# - - - LOGGER FUNCTIONS - - -


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

    for i in range(len(individual_error_list)):
        individual_error_list[i] = individual_error_list[i] / num_of_individuals[i]

    return individual_correct_list, correct


def test_logs(report, conf_matrix, filename):
    with open(filename, 'w') as plik:
        plik.write("WIELKOSCI Z CZESCI TESTOWEJ\n\nRaport\n")
        plik.write(report)

        plik.write("\nMacierz pomylek\n")
        plik.write(str(conf_matrix))


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


# - - - PLOT FUNCTION- - -

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()
