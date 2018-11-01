from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

df1 = pd.read_csv("./test_data.csv", header=None)
df2 = pd.read_csv("./train_data.csv", header=None)
df3 = pd.read_csv("./train_labels.csv", header=None)
df4 = pd.read_csv("./best.csv")

y_best = df4['Sample_label'].values

# 72,82,83,84,85,86,87,88,89,
# df1 = df1.drop(columns=[72, 73, 76, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,216, 217, 218, 219, 228,229, 230, 231, 232, 233, 234,235, 236,237,238,239 ])
# df2 = df2.drop(columns=[72, 73, 76, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,216, 217, 218, 219, 228,229, 230, 231, 232, 233, 234,235, 236,237,238,239 ])

# good predictors
# 228,229, 255

#23, 47, 119, 143
df2 = df2.drop(columns=[
    71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219,
])
df1 = df1.drop(columns=[
    71, 72, 88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219,
])


df2 = df2.drop([1558, 3820, 3411, 1842, 239, 4308, 3008, 3119, 1763, 4296, 3423, 1073, 3650])
df3 = df3.drop([1558, 3820, 3411, 1842, 239, 4308, 3008, 3119, 1763, 4296, 3423, 1073, 3650])

y = df3.values
# for i, x in enumerate(y.T[0]):
#     print("{0}: {1}".format(i, x))


def get_scores(X_train, y_train, n_comp=3):
    pca = PCA(n_components=n_comp, svd_solver='full')
    scores = pca.fit_transform(X_train, y_train.flatten())
    print(pca.explained_variance_ratio_)
    return scores


def get_best_features(df, y):
    rythm_data = df.loc[:, 0:168]
    chroma_data = df.loc[:, 168:168 + 48]
    mfcc_data = df.loc[:, 168 + 48:]
    score_matrix = get_scores(rythm_data.loc[:, : 24 - 1], y)

    print("start")
    for i in range(1, 7):
        df = rythm_data.loc[:, i * 24:(i + 1) * 24 - 1]
        score_matrix = np.hstack((score_matrix, (get_scores(df.values, y, 10))))
    print("start")
    for i in range(4):
        df = chroma_data.loc[:, 168 + i * 12: 168 + (i + 1) * 12 - 1]
        score_matrix = np.hstack((score_matrix, (get_scores(df.values, y, 8))))
    print("start")
    for i in range(4):
        df = mfcc_data.loc[:, 168 + 48 + i * 12:168 + 48 + (i + 1) * 12 - 1]
        score_matrix = np.hstack((score_matrix, (get_scores(df.values, y, 8))))
    print("end")

    return score_matrix


min_max_scaler = preprocessing.MinMaxScaler()
normalizer_scaler = preprocessing.Normalizer()
max_abs_scaler = preprocessing.MaxAbsScaler()
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)  # 0.64951
standard_scaler = preprocessing.StandardScaler()
robust = preprocessing.RobustScaler()
poly = preprocessing.PolynomialFeatures(2)

# df2 = (df2 - df2.mean()) / (df2.max() - df2.min())
# df1 = (df1 - df1.mean()) / (df1.max() - df1.min())

X = df2.values  # get_best_features(df2, y)
X_t = df1.values  # get_best_features(df1, y)

quantile_transformer.fit(X)  # best
X = quantile_transformer.transform(X)  # best
X_t = quantile_transformer.transform(X_t)  # best
#
# rythm_data = X[:, 0:168]
# chroma_data = X[:, 168:168 + 48]
# mfcc_data = X[:, 168 + 48:]


# rythm = rythm_data[:, 0:24]
# print(rythm_data.shape)
# print(chroma_data.shape)
# print(mfcc_data.shape)
# chroma = chroma_data[:, :36]
# mfcc = mfcc_data[:, :4]

# w = [1]*len(y)
#
# print(len(y))
import collections

amounts = collections.Counter(y.flatten())
print(amounts)


# for i in range(len(y)):
#     w[i] = 1 - amounts.get(y.flatten()[i])/len(y)
# cw = {
#         1: 1-2178/len(y),
#         2: 1-618/len(y),
#         3: 1-326/len(y),
#         6: 1-260/len(y),
#         4: 1-253/len(y),
#         5: 1-214/len(y),
#         8: 1-195/len(y), 7: 1-141/len(y), 9: 1-92/len(y), 10: 1-86/len(y)}
# print(w)

def pred(X, y, state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=state)
    y_pred, score, prob = classifier(X_train, y_train, X_test, y_test)
    print("Prediction score {0}: {1:.3}".format(state, score))
    return y_pred, score


def crossValidation(X, y, runs=3):
    accs = []
    for i in range(runs):
        ys, ac = pred(X, y, i)
        accs.append(ac)
    print("mean score: " + str(np.mean(accs)))


def predf(X_train, y_train, X_test):
    lg = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=5000, tol=1e-6)
    lg.fit(X_train, y_train.flatten())
    y_pred = lg.predict(X_test)
    return y_pred


def classifier(X_train, y_train, X_test, y_test, state=0):
    # clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=1000, tol=1e-6) #best
    # clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000, tol=1e-6, n_jobs=10) # best loss
    # clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=10000, tol=1e-6, n_jobs=10)
    # clf = MLPClassifier(alpha=1)
    clf = MLPClassifier(alpha=1, random_state=state)
    clf.fit(X_train, y_train.flatten())
    y_pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    return y_pred, score, prob


def classify(X_train, y_train, X_test, y_test):
    y_pred, score, prob = classifier(X_train, y_train, X_test, y_test)
    print("Accuracy compared to best prediction: {0}".format(score))
    return y_pred, prob


# pred_np1 = np.array(preds)
# print(pred_np1.shape)
# print(y_test.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(pred_np1.T, y_test, test_size=0.33, random_state=1)
#
# predf2(X_train, y_train, X_test, y_test)
# p = pred(X, y)


def out(X_train, y, X_test, y_best):
    prediction, prob = classify(X_train, y, X_test, y_best)
    probs = pd.DataFrame(data=prob)
    probs.columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7',
                     'Class_8', 'Class_9', 'Class_10', ]
    probs.index.name = 'Sample_id'
    probs.index += 1
    probs.to_csv('loss.csv')
    print("Wrote loss.csv")
    dataframe = pd.DataFrame(data=np.array([range(1, len(prediction) + 1), prediction]).T)
    dataframe.columns = ['Sample_id', 'Sample_label']
    dataframe.to_csv('out.csv', index=False)
    print("Wrote out.csv")


crossValidation(X, y, 10)
out(X, y, X_t, y_best)
