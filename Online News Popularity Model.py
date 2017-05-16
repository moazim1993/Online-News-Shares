# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from scipy.stats import norm


# Part 1 Exploritory Data Analysis
# On the training set get discription, distributions
def show_Distribution(train):  # input data, output distribution and discript
    for i in train:
        info = train[i].describe()[1:-2]
        print(info)
        if len(train[i].unique()) < 20:
            num_bins = len(train[i].unique())
        else:
            num_bins = 20
        bins = np.linspace(min(train[i]), max(train[i]), num=num_bins)
        plt.hist(train[i], bins, histtype='bar', rwidth=.8)
        plt.title("Historgams\nnumbins: "+str(num_bins))
        plt.xlabel(i)
        plt.ylabel('count')
        plt.legend()
        plt.show()
        plt.pause(1)
        plt.clf()


# input training data
def show_Correl(train):  # output correl coef heatmap
    corr = np.corrcoef(np.array(train.values).T)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.show()


#  Part 2 feature selection
#  input numpy arrays train and test, and number of components
def do_PCA(trainX, testX, n):  # output all data fitted to train X w PCA
    pipeline = Pipeline([('scaling', StandardScaler()),
                         ('pca', PCA(n_components=n))])
    pipeline.fit(trainX)
    new_X = pipeline.transform(trainX)
    new_testX = pipeline.transform(testX)
    pca = pipeline.named_steps['pca']
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis()
    plt.title("PCA Variance")
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')
    plt.show()
    return new_X, new_testX


# input numpy array train and test,and number of components
def do_ICA(trainX, testX, n=None):  # output all data fitted to the train X
    if n is None:
        accuracies = []
        tst_accuracies = []
        for n in range(1, len(train.T)):  # incremental adding of components
            pipeline = Pipeline([('scaling', StandardScaler()),
                                 ('ica', FastICA(n_components=n, tol=0.01))])
            pipeline.fit(trainX)
            new_X = pipeline.transform(trainX)
            new_testX = pipeline.transform(testX)
            clf = LinearSVC()
            clf.fit(new_X, y)
            accuracy = clf.score(new_X, y)
            tst_acuracy = clf.score(new_testX, tst_y)
            # print(accuracy, tst_acuracy)
            accuracies.append(accuracy)
            tst_accuracies.append(tst_acuracy)

        print(n, " Components")
        plt.plot(accuracies, label='train')
        plt.plot(tst_accuracies, label='test')
        plt.title("ICA Components to Performance")
        plt.legend()
        plt.ylabel("accuracy")
        plt.xlabel("number of components")
        plt.show()
        return new_X, new_testX
    else:
        pipeline = Pipeline([('scaling', StandardScaler()),
                             ('ica', FastICA(n_components=n, tol=0.01))])
    pipeline.fit(trainX)
    new_X = pipeline.transform(trainX)
    new_testX = pipeline.transform(testX)
    return new_X, new_testX


# input train and test data
def do_SVM(X, y, tstX, tstY):  # output dif accuracy for SVM
    results = [["slack strength", "kernal type",
               "traing accuracy", "test accuracy"]]
    Kernal_type = ['rbf', 'linear']
    # l2_strenght 100 being least regularization .01 being higest
    l2_strenght = [10, 1, .5, .1, .01]
    for c in l2_strenght:  # loop to optimize hyper params
        for k in Kernal_type:
            # print(c, k)
            clf = SVC(kernel=k, C=c)
            clf.fit(X, y)
            train = clf.score(X, y)
            # print("svm train :", train)
            test = clf.score(tstX, tstY)
            # print("svm train :", test)
            result = [c, k, train, test]
            results.append(result)
    return results


def do_GDANaiveBayes(X, Y, tstX, tstY, prior=None):
        # prior P(Y=yi) is same across both i
    if prior is None:
        prob_y1 = sum(Y)/len(Y)
        prob_y0 = 1 - prob_y1
    elif prior == '=':
        prob_y0, prob_y1 = .5, .5
    else:  # laplace smoothing
        prob_y1 = sum(Y)+1/len(Y)+prior
        prob_y0 = (len(Y)-sum(Y))+1/len(Y)+prior
    data = np.concatenate((X, Y.reshape(len(Y), 1)), axis=1)
    data0 = []
    data1 = []
    for row in data:
        if row[-1] == 0:
            data0.append(row)
        else:
            data1.append(row)

    data0 = np.array(data0)
    data1 = np.array(data1)
    means_y0 = [np.mean(x) for x in data0[:, :-1].T]
    means_y1 = [np.mean(x) for x in data1[:, :-1].T]
    std_y0 = [np.std(x) for x in data0[:, :-1].T]
    std_y1 = [np.std(x) for x in data1[:, :-1].T]
    # calculated training accuracy
    yhat = []
    for x in X:
        yhat0 = np.log(prob_y0)
        yhat1 = np.log(prob_y1)
        for i in range(len(x)):  # log pdf to compound easier
            yhat0 += norm(means_y0[i], std_y0[i]).logpdf(x[i])
            yhat1 += norm(means_y1[i], std_y1[i]).logpdf(x[i])
        if yhat0 > yhat1:
            yhat.append(0)
        else:
            yhat.append(1)
    accuracy = []
    for i, j in zip(Y, yhat):
        if i == j:
            accuracy.append(1)
        else:
            accuracy.append(0)
    accuracy1 = (sum(accuracy)/len(accuracy))
    print(accuracy1)
    # calculated test accuracy
    yhat = []
    for x in tstX:
        yhat0 = np.log(prob_y0)
        yhat1 = np.log(prob_y1)
        for i in range(len(x)):  # log pdf to compound easier
            yhat0 += norm(means_y0[i], std_y0[i]).logpdf(x[i])
            yhat1 += norm(means_y1[i], std_y1[i]).logpdf(x[i])
        if yhat0 > yhat1:
            yhat.append(0)
        else:
            yhat.append(1)
    accuracy = []
    for i, j in zip(tstY, yhat):
        if i == j:
            accuracy.append(1)
        else:
            accuracy.append(0)
    accuracy2 = (sum(accuracy)/len(accuracy))
    print(accuracy2)
    return accuracy1, accuracy2

# _______________________MAIN_________________
# part1
# Get data, split into train and test and save for reproducibility
df = pd.read_csv('OnlineNewsPopularity.csv', sep=',')
df.drop(df.columns[0], axis=1, inplace=True)  # remove url
train = df.sample(frac=0.7, random_state=200)
test = df.drop(train.index)
X, y = train.drop(" shares", axis=1), train[' shares']
testX, testY = test.drop(" shares", axis=1), test[' shares']
# get a sense for distrib and data set
show_Distribution(train)
show_Correl(train)
plt.clf()
# part2 feature selection
train, test = np.array(train), np.array(test)
# print(np.shape(train), np.shape(test))  # debug/check
# delete pd df index and cols to format for np array
train, test = train[1:, 1:], test[1:, 1:]
X, y = train[:, :-1], train[:, -1]
tst_X, tst_y = test[:, :-1], test[:, -1]
# binarize Y based on median, create a (50/50) binary classification
y = [1 if i > 1400 else 0 for i in y]
tst_y = [1 if i > 1400 else 0 for i in tst_y]
do_PCA(X, tst_X, 50)
plt.clf()
# 16 PCA Commponents picked based of results
fset1_X, fset1_tstX = do_PCA(X, tst_X, 16)
plt.clf()
# all picked based of ICA perfom. results
fset2_X, fset2_tstX = do_ICA(X, tst_X)
# part 3 modeling
svm_results_f1 = do_SVM(fset1_X, y, fset1_tstX, tst_y)
svm_results_f2 = do_SVM(fset2_X, y, fset2_tstX, tst_y)
print('PCA data\n', svm_results_f1, '\n', 'ICA data\n', svm_results_f2)
training_f1, test_f1 = do_GDANaiveBayes(fset1_X, y, fset1_tstX, tst_y)
print(training_f1, test_f1)
