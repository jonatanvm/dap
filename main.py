from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import pandas as pd
from sklearn import preprocessing

df1 = pd.read_csv("./test_data.csv", header=None)
df2 = pd.read_csv("./train_data.csv", header=None)
df3 = pd.read_csv("./train_labels.csv", header=None)
print(df1)


def drop(num):
    return df1.drop(columns=[num]), df2.drop(columns=[num])


df2 = df2.drop(columns=[
    72, 90, 91, 92, 93, 94, 95
])
df2 = df2.drop(columns=[
    216, 217, 218, 219
])

df2 = df2.drop(columns=[168 + 48 + 13])
# df2 = df2.drop(columns=[168 + 48 + 14])

# # # chroma
# for i in range(11):
#     df2 = df2.drop(columns=[168+i])
#
#
# for i in range(2):
#     df2 = df2.drop(columns=[168+48+13+i])


X = df2.values
y = df3.values

# quantile_transformer = preprocessing.QuantileTransformer(random_state=0)  # 0.64951
# X = quantile_transformer.fit_transform(X)

rythm_data = df2.loc[:, 0:168].values
rythm_data = rythm_data[:, 0:24]
print(rythm_data.shape)
chroma_data = df2.loc[:, 168:168 + 48].values
print(chroma_data.shape)
mfcc_data = df2.loc[:, 168 + 48:].values
print(mfcc_data.shape)
# rythm_data = np.delete(rythm_data, [95], 1)
# mfcc_data = np.delete(mfcc_data, [13], 1)

# X_test = df1.values
X_train = rythm_data
# y_test = df3.values
# y_train = df3.values

min_max_scaler = preprocessing.MinMaxScaler()
# X_train = np.nan_to_num(np.log(X_train))
print(np.isnan(X_train).any())
pca = PCA(n_components=5, svd_solver='full')
res = pca.fit(X_train)
scores = pca.transform(X_train)
loading = pca.components_
print(X_train.shape)
print(scores.shape)
print(pca.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2)

ax = axes[0]

ax.scatter(loading[0], loading[1])
print(X_train.shape[1])
labels = df2.columns
for i in range(X_train.shape[1]):
    ax.annotate(str(labels[i]), (loading[0][i], loading[1][i]))

ax = axes[1]
ax.scatter(loading[1], loading[2])

for i in range(X_train.shape[1]):
    ax.annotate(labels[i], (loading[1][i], loading[2][i]))

plt.show()
