import tf

from functions import *

# - - - DATA PREPARATION - - -

df = pd.read_csv('data.csv')

df.iloc[:, -1] = df.iloc[:, -1] - 1

df.to_csv('data_fixed.csv', index=False)

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

test.to_csv('test.csv', index=False)

train, x_train, y_train = oversample_set(train, True)
valid, x_valid, y_valid = oversample_set(valid)
test, x_test, y_test = oversample_set(test)

# - - - SVM - - -

svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)

y_pred1 = svm_model.predict(x_test)

print(classification_report(y_test, y_pred1))

# - - - NEURAL NETWORK - - -

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

y_pred2 = nn_model.predict(x_test)

# print(classification_report(y_test, y_pred2))

plot_accuracy(history)
plot_loss(history)

types_list = []
for i in y_test:
    if i not in types_list:
        types_list.append(i)

print(y_pred2)
y_pred_bin = prepare_type_list(y_pred2)
print(y_pred_bin)

report = classification_report(y_test, y_pred1)
conf_matrix = confusion_matrix(y_test, y_pred1)

global_error = calculate_global_error(y_test, y_pred1)
individual_error_list, individual_correct_list, correct = calculate_individual_error(y_test, y_pred1, types_list)


file1 = "testing_logs.txt"
file2 = "testing_results.txt"
test_logs(global_error, individual_error_list, report, conf_matrix, file1)
result_logs(y_test, y_pred1, individual_correct_list, correct, file2)

report1 = classification_report(y_test, y_pred_bin) # TODO moze trzeba dac y_pred2
conf_matrix1 = confusion_matrix(y_test, y_pred_bin)
global_error1 = calculate_global_error(y_test, y_pred_bin)
individual_error_list1, individual_correct_list1, correct1 = calculate_individual_error(y_test, y_pred_bin, types_list)
file3 = "testing_logs1.txt"
file4 = "testing_results1.txt"
test_logs(global_error1, individual_error_list1, report1, conf_matrix1, file3)
result_logs(y_test, y_pred_bin, individual_correct_list1, correct1, file4)
