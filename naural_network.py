from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

df1 = pd.read_csv("./test_data.csv", header=None)
df2 = pd.read_csv("./train_data.csv", header=None)
df3 = pd.read_csv("./train_labels.csv", header=None)
df4 = pd.read_csv("./best.csv")
df4 = df4['Sample_label'].values
df2 = df2.drop(columns=[
    71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219
])
df1 = df1.drop(columns=[
    71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219
])
min_max_scaler = preprocessing.MinMaxScaler()
normalizer_scaler = preprocessing.Normalizer()
max_abs_scaler = preprocessing.MaxAbsScaler()
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
standard_scaler = preprocessing.StandardScaler()

X = df2.values
X_t = df1.values
y = df3.values

X = standard_scaler.fit_transform(X)
X_t = standard_scaler.fit_transform(X_t)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(260, 150, 100, 50, 25, 10, 5),
                    random_state=1)

clf.fit(X, y.flatten())

# predict = clf.predict(X_test)

print(clf.score(X_t, df4.flatten()))
