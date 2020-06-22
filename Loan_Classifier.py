# LDA, QDA, Logistic Regression, and k-Nearest Neighbors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlwings as xw
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.ensemble as skle
import sklearn.neighbors as skl_nb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import seaborn as sns
# 3 models for selecting variables
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import f_classif
from sklearn import preprocessing
import sklearn.model_selection as skl_ms
from sklearn.metrics import confusion_matrix
# create file in notepad and see if it can read that!
from xlrd import *
import win32com.client
import csv
import sys

#This classifies whether a loan will default based on a set of parameters

# Unfortunately I am not able to upload this data....
df = pd.read_excel(r"C:\Users\user\Documents\sample_data.xlsx", sep='\s*,\s*',
                   dtype={'ID': str})  # about 40 variables, many dependent on each other

max_nans = len(df) * 0.5  # only 201 instances when only dropping features missing 80% of data
df = df.copy().drop(columns=['ID'])




df = df.loc[:, (df.isnull().sum(axis=0) <= max_nans)]  # dropping features involved in less than 50% of data points
# make half the examples cases of default and the other half non-default cases

df = df.dropna().reset_index(drop=True)  # removing all the indices without values

# changing type to boolean
type = df['TYPE'].copy()
type[type == 'With financials'] = 1
type[type == 'Without financials'] = 0
df['TYPE'] = type
df_default = df.where(df['DEFAULT_FLG'] == 1)  # replaces all false values with nan
df_default = df_default.dropna()

df_non_default = df.where(df['DEFAULT_FLG'] == 0).sample(len(df_default),
                                                         random_state=23)  # sampling as many defaults as non-defaults.
df_non_default = df_non_default.dropna()

frames = [df_default, df_non_default]

df2 = df  # we will be using the regular dataset and weighing in the model fitting
X = df2.copy().drop(columns=['DEFAULT_FLG'])
X = preprocessing.normalize(X)  # standardizing

y = df2['DEFAULT_FLG']
corr = np.abs(np.corrcoef(X,
                          rowvar=False))  # correlation matrix
tri = np.triu(corr)
np.fill_diagonal(tri, 0)  # we can't let the variables' correlation with selves ruin things
to_drop = np.argwhere(tri > 0.8)  # you can't let it include the diagonal

X = np.delete(X, to_drop[:, 1],
              axis=1)  # drops every column exceeding this correlation and this shouldn't drop both correlated
X = pd.DataFrame(X)


(tn, fp, fn, tp) = (0, 0, 0, 0)  # confusion vector

#
model_LDA = skl_da.LinearDiscriminantAnalysis()  # https://sci2s.ugr.es/keel/pdf/specific/articulo/xue_do_2008.pdf
# ^perhaps we can't balance
model_QDA = skl_da.QuadraticDiscriminantAnalysis()

model_LR = skl_lm.LogisticRegression(multi_class='ovr', class_weight='balanced')  # ovr means binary problem
model_boostA = skle.AdaBoostClassifier()



# principal component analysis worked best of our parameter selection
pca = PCA(n_components=10)#reducing parameter space to 10 with principal component analysis
fit = pca.fit(X)

X = fit.transform(X)
X = pd.DataFrame(X)


model_0s = np.zeros(len(df))

error_0s = 0
print(f"X dimensions:{X.shape}")



# optimizing tree
max_val = 22

error_boostA = 0
# minimum values for tree
# computes cross-validation error
n_split = 10


# would be quicker if we did everything in a loop so we wouldn't have to split the datasets but the code looks nicer
# class CV:
def CV(model, n_split, X, Y):
    # takes the arguments model and n_splits for kfold error estimation
    cross_val = skl_ms.KFold(n_splits=n_split, shuffle=True, random_state=2)
    model_err = 0
    (tn, fp, fn, tp) = (0, 0, 0, 0)
    for train, test in cross_val.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        Y_train, Y_test = Y.iloc[train], Y.iloc[test]
        model.fit(X_train, Y_train)
        pred_model = model.predict(X_test)
        (tn, fp, fn, tp) = (tn, fp, fn,
                            tp) + confusion_matrix(
            Y_test, pred_model).ravel() / n_split
        model_err += np.mean(pred_model != Y_test)
    model_err /= n_split
    sum_confusion = tn + fp + fn + tp
    tn = tn / sum_confusion
    tp = tp / sum_confusion
    fp = fp / sum_confusion
    fn = fn / sum_confusion
    print(f"tn:{tn}, fp:{fp},fn:{fn},tp:{tp}")  # we will check this out for the best mode
    return model_err


# classification trees

# LR, LDA, and QDA are done differently so we can tweak probabilities
# if min_samp == 2:


error_LR = CV(model_LR, n_split, X, y)
error_LDA = CV(model_LDA, n_split, X, y)
error_QDA = CV(model_QDA, n_split, X, y)

# # classification and regression trees
index = 0  # counting index for CART
max_val = 22
min_values = np.arange(2, max_val, 1)
error_cart = np.zeros(len(min_values))
for min_samp in min_values:
    print(min_samp)
    model_cart = RandomForestClassifier(min_samples_split=min_samp, class_weight='balanced')
    error_cart[index] = CV(model_cart, n_split, X, y)
    index += 1


#boosting algorithms
error_boostA = CV(model_boostA, n_split, X, y)
error_boostG = np.zeros((5, 5))
for j in range(0, 5, 1):  # max_depth
    print(j + 1)
    for i in range(0, 5, 1):  # min_samples_leaf
        print(f"Maximum depth:{j + 1}, Minimum sample leaves:{i + 1}")
        model_boostG = skle.GradientBoostingClassifier(max_depth=j + 1,
                                                       min_samples_leaf=i + 1)  # I don't know how to experiment here
        error_boostG[i][j] = CV(model_boostG, n_split, X, y)

max_k = 50
error_k = np.zeros(max_k)
for k in range(0, 50, 1):  # last iteration is at 50
    print(k + 1)
    model_k = skl_nb.KNeighborsClassifier(n_neighbors=k + 1)
    error_k[k] = CV(model_k, n_split, X, y)


print(f"error_LDA:{error_LDA}")


print(f"error_QDA:{error_QDA}")


print(f"Logistic Regression error:{error_LR}")



plt.plot(min_values, error_cart)
plt.title("CART Error Plot")
plt.xlabel("Minimum samples for split")  # min samples required to split a node, i.e. we were not overfitting
plt.ylabel("Missclassification Error")
plt.show()
min_CART = np.min(error_cart)
min_tree = np.where(error_cart == min_CART)  # k that achieved lowest value
print(f"Minimum error_CART:{min_CART} for minimum trees=:{min_tree}")

plt.plot(error_boostG)
plt.title("Boosting Error Plot")
plt.show()
min_boostG = np.min(error_boostG)
min_components = np.where(error_boostG == min_boostG)  # k that achieved lowest value
print(
    f"Minimum error for gradient boosting:{min_boostG} for min_samples_leaf and max_depth=:{min_components} respectively")

plt.plot(error_k)
plt.title("k-NN Error Plot")
plt.xlabel("Number of Neighbors")
plt.ylabel("Error")
plt.show()
min_nb = np.min(error_k)
min_k = np.where(error_k == min_nb)  # k that achieved lowest value
print(f"Minimum error for k-NN:{min_nb} for k=:{min_k}")

print(f"Gradient boosting error:{error_boostG.model_err}")
print(f"Adaboost error:{error_boostA}")

# plt.plot(error_k)
# plt.show()


# plt.plot(error_cart.model_err)
# plt.show()
# min_cart = np.min(error_cart.model_err)
# min_trees = np.where(error_cart.model_err == min_cart)  # k that achieved lowest value
# print(f"error_cart:{min_cart} for number of trees=:{min_trees}")
# print(f"tn:{tn}, fp:{fp},fn:{fn},tp:{tp}")
