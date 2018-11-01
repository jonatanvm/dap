from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, log_loss
import random

df1 = pd.read_csv("./test_data.csv", header=None)
df2 = pd.read_csv("./train_data.csv", header=None)
df3 = pd.read_csv("./train_labels.csv", header=None)
df4 = pd.read_csv("./best.csv")
y_best = df4['Sample_label'].values
mat = df3.values.flatten()
print(mat)


def i_for(i):
    return np.where(mat == i)


# 72,82,83,84,85,86,87,88,89,
df2 = df2.drop(columns=[71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219,
 ])

df1 = df1.drop(columns=[71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219,
])
# df1 = df1.drop(columns=[72, 73, 76,82,83,84, 85,87, 88, 89, 90, 91, 92, 93, 94, 95])
# 1
# df1 = df1.drop(columns=[72, 73, 76, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, ])
# 72
# 71
# 2


# 3
# 228,229, 230, 231, 232, 233, 234,235, 236,237,238,239
# df2 = df2.drop(columns=[216, 217, 218, 219, 228, 229, 230, 231,232, 233,234,235,236])
# df1 = df1.drop(columns=[229,230])


rythm_data = df2.loc[:, 0:168 - 1]
chroma_data = df2.loc[:, 168:168 + 48 - 1]
mfcc_data = df2.loc[:, 168 + 48:]
rythm_data_2 = df2.loc[:, 0:168].values
chroma_data_2 = df2.loc[:, 168:168 + 48].values
mfcc_data_2 = df2.loc[:, 168 + 48:].values

fig, axes = plt.subplots(2, 2,figsize=(15,15))
axes[0, 0].plot(rythm_data.values.T)
axes[0, 1].plot(chroma_data.values.T)
axes[1, 0].plot(mfcc_data.values.T)
# rythm = rythm_data[:, 0:24]
# print(rythm_data.shape)
# print(chroma_data.shape)
# print(mfcc_data.shape)
# chroma = chroma_data[:, :36]
# mfcc = mfcc_data[:, :4]
#
# np.random.seed(1)
# indicies = np.random.choice(X.shape[0], 20)
# for i in range(X.shape[0]):
#     plt.plot(rythm_data[i, 0:90])

# l = [1,2,3,4,5,6,7,8,9,10]
# for i in l:
#     plt.figure()
#     plt.plot(mfcc_data.iloc[i_for(i)].T)

# max = 0
# v = np.max(rythm_data, axis=1)
# print(v)
#
# print(v.shape)
# print(np.max(v))
# print(np.argmax(v))
# plt.figure()
# plt.plot(rythm_data_2.T)
# for i in np.random.choice(X.shape[0], 500):
#     plt.plot(chroma_data[i, :161])

# plt.figure()
# for i in np.random.choice(X.shape[0], 500):
#     plt.plot(mfcc_data[i, :161])
# plt.legend(loc='best')
plt.show()
