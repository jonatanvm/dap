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
mat = df3.values.flatten()
print(mat)


def i_for(i):
    return np.where(mat == i)


# print(df1)


def drop(num):
    return df1.drop(columns=[num], inplace=True), df2.drop(columns=[num], inplace=True)


X = df2.values
y = df3.values


# quantile_transformer = preprocessing.QuantileTransformer(random_state=0)  # 0.64951
# X = quantile_transformer.fit_transform(X)

def test_rythm(data):
    for i in range(6,7):
        print(i)
        labels = data.loc[:, i * 24:(i + 1) * 24 - 1].columns
        print(labels)
        f1 = data.loc[:, i * 24:(i + 1) * 24 - 1].values
        print(f1.shape)
        scores, loadings = run_pca(f1)
        plot_loadings(f1, loadings, labels)


def test_chroma(data):
    for i in range(4):
        print(i)
        labels = data.loc[:, 168 + i * 12:168 + (i + 1) * 12 - 1].columns
        print(labels)
        f1 = data.loc[:, 168 + i * 12:168 + (i + 1) * 12 - 1].values
        print(f1.shape)
        scores, loadings = run_pca(f1)
        plot_loadings(f1, loadings, labels)


def test_mfcc(data):
    for i in range(4):
        print(i)
        labels = data.loc[:, 168 + 48 + i * 12:168 + 48 + (i + 1) * 12 - 1].columns
        print(labels)
        f1 = data.loc[:, 168 + 48 + i * 12:168 + 48 + (i + 1) * 12 - 1].values
        print(f1)
        scores, loadings = run_pca(f1)
        plot_loadings(f1, loadings, labels)


def run_pca(X_train):
    pca = PCA(n_components=5, svd_solver='full')
    scores = pca.fit_transform(X_train)
    loading = pca.components_
    print(pca.explained_variance_ratio_)
    return scores, loading


def plot_loadings(X, loading, lab_pos):
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]

    # ax.scatter(loading[0], loading[1])
    for i in range(loading.shape[1]):
        ax.plot([0, loading[0][i]], [0, loading[1][i]])

    for i in range(X.shape[1]):
        ax.annotate(str(lab_pos[i]), (loading[0][i], loading[1][i]))

    ax = axes[1]
    # ax.scatter(loading[1], loading[2])
    for i in range(loading.shape[1]):
        ax.plot([0, loading[1][i]], [0, loading[2][i]])

    for i in range(X.shape[1]):
        ax.annotate(str(lab_pos[i]), (loading[1][i], loading[2][i]))


def plot_scores(X, scores, lab_pos):
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]

    ax.scatter(scores[0], scores[1])

    print(scores.shape)
    # for i in range(X.shape[1]):
    #     ax.annotate(str(lab_pos[i]), (scores[0][i], scores[1][i]))

    ax = axes[1]
    ax.scatter(scores[1], scores[2])

    # for i in range(X.shape[1]):
    #     ax.annotate(lab_pos[i], (scores[1][i], scores[2][i]))


# drop(95)
# drop(94)
# drop(93)
# drop(92)
# drop(91)
# drop(90)
# drop(72)
# drop(89)
# drop(88)
# drop(71)
# drop(70)
# 2
# drop(192)
# drop(193)
# 3
# drop(241)
# drop(243)
drop(216)
drop(217)
drop(218)
drop(219)

# drop(228)
# drop(239)#
# drop(240)
# drop(252)

drop(23)
drop(47)
drop(71)
drop(72)
drop(92)
drop(93)
drop(94)
drop(95)
drop(119)
drop(143)
# drop(167)



rythm_data = df2.loc[:, 0:167]
chroma_data = df2.loc[:, 168:168 + 47]
mfcc_data = df2.loc[:, 168 + 48:]

pd.set_option('display.max_rows', None)  # or 1000
l = [1,2,3,4,5,6,7,8,9,10]
for i in l:
    test_rythm(rythm_data.iloc[i_for(i)])
    # test_chroma(chroma_data.iloc[i_for(i)])
    # test_mfcc(mfcc_data.iloc[i_for(i)])
# test_rythm(rythm_data)
# test_chroma(chroma_data)
# test_mfcc(mfcc_data)
plt.show()
