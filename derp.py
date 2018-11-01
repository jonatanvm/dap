from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
df1 = pd.read_csv("./test_data.csv", header=None)
df2 = pd.read_csv("./train_data.csv", header=None)
df3 = pd.read_csv("./train_labels.csv", header=None)
df4 = pd.read_csv("./best.csv")
y = df3.values
y_best = df4['Sample_label'].values

## Preprocessing

# Data selection
df2 = df2.drop(columns=[
    71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219,
])
df1 = df1.drop(columns=[
    71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219,
])

# Scaling
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)

X = df2.values
X_t = df1.values
quantile_transformer.fit(X)
X = quantile_transformer.transform(X)
X_t = quantile_transformer.transform(X_t)


def predLog(X, y, state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=state)
    y_pred, score, prob = classifier(X_train, y_train, X_test, y_test)

    print("Prediction score {0}: {1}".format(state, score))
    return y_pred, score


def crossValidation(X, y):
    accs = []
    for i in range(10):
        ys, ac = predLog(X, y, i)
        accs.append(ac)
    print("Mean score: " + str(np.mean(accs)))


def classifier(X_train, y_train, X_test, y_test):
    clf = MLPClassifier(alpha=4)
    clf.fit(X_train, y_train.flatten())
    y_pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    return y_pred, score, prob


def predMLP(X_train, y_train, X_test, y_test):
    y_pred, score, prob = classifier(X_train, y_train, X_test, y_test)
    print("Accuracy compared to best prediction: {0}".format(score))
    return y_pred, prob


# Validation
# crossValidation(X, y)
prediction, prob = predMLP(X, y, X_t, y_best)