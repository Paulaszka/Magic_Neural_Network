from functions import *

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv('magic04.data', names=cols)

df["class"] = (df["class"] == "g").astype(int)

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

train, x_train, y_train = oversample_set(train, True)
valid, x_valid, y_valid = oversample_set(valid)
test, x_test, y_test = oversample_set(test)

svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

print(classification_report(y_test, y_pred))


