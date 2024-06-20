from sklearn.svm import SVC
import tensorflow as tf
import pandas as pd
from sklearn.metrics import *
from functions import *

# - - - DATA PREPARATION - - -

data = pd.read_csv('magic04.data')

mapping = {'g': 0, 'h': 1}
data = data.replace(mapping).infer_objects(copy=False)

# caly zbior danych liczy 19020 instancji, czesc treningowa 13314, a czesc testowa 5706
train, test = np.split(data.sample(frac=1), [int(0.7 * len(data))])

train, x_train, y_train = oversample_set(train, True)
test, x_test, y_test = oversample_set(test)

# - - - SVM - - -

svm_model = SVC(kernel='linear')
svm_model = svm_model.fit(x_train, y_train)

y_pred1 = svm_model.predict(x_test)

# - - - NEURAL NETWORK - - -

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(x_train, y_train, epochs=600, batch_size=951, verbose=1)

y_pred2 = nn_model.predict(x_test)

plot_loss(history)

# - - - RAPORT CALCULATIONS - - -

types_list = []
for i in y_test:
    if i not in types_list:
        types_list.append(i)

y_pred_bin = []
for i in range(len(y_pred2)):
    if y_pred2[i] >= 0.5:
        y_pred_bin.append(1)
    else:
        y_pred_bin.append(0)

report = classification_report(y_test, y_pred1)
conf_matrix = confusion_matrix(y_test, y_pred1)


individual_correct_list, correct = calculate_individual_error(y_test, y_pred1, types_list)
test_logs(report, conf_matrix, "svm_testing_logs.txt")
result_logs(y_test, y_pred1, individual_correct_list, correct, "svm_testing_results.txt")


individual_correct_list1, correct1 = calculate_individual_error(y_test, y_pred_bin, types_list)
test_logs(report, conf_matrix, "mlp_testing_logs.txt")
result_logs(y_test, y_pred_bin, individual_correct_list1, correct1, "mlp_testing_results.txt")
