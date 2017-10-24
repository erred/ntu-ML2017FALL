import csv

import numpy as np

col_filter = []
col_filter.append(list(range(0, 1)))  # age
col_filter.append(list(range(1, 2)))  # fnlwgt
# col_filter.append(list(range(2, 3)))  # sex
col_filter.append(list(range(3, 5)))  # capital gain/loss
col_filter.append(list(range(5, 6)))  # hours_per_week
col_filter.append(list(range(6, 15)))  # employer
col_filter.append(list(range(15, 22)))  # edu_num
col_filter.append(list(range(22, 31)))  # edu
# col_filter.append(list(range(31, 38)))  # maritial
col_filter.append(list(range(38,53))) # occupation
col_filter.append(list(range(53, 59)))  # relationship
# col_filter.append(list(range(59,64))) # race
col_filter.append(list(range(64, 106)))  # country
col_filter = sum(col_filter, [])
FEATURES = len(col_filter)


def activate(x):
    out = 1.0 / (1.0 + np.exp(x))
    return np.clip(out, 1e-8, 1-(1e-8))

def saver(weights, bias, modelFile):
    with open(modelFile, 'w') as of:
        w = csv.writer(of)
        w.writerow([bias])
        w.writerow(weights.tolist())

def loader(modelFile):
    with open(modelFile, 'r') as of:
        r = csv.reader(of)
        row = next(r)
        bias = float(row[0])
        row = next(r)
        weights = np.array([float(x) for x in row])
    return weights, bias

def saveOutput(arr, outputFile):
    with open(outputFile, 'w') as of:
        w = csv.writer(of)
        w.writerow(["id", "label"])
        w.writerows(arr)

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def readData(X_train, X_test, Y_train, col_filter=col_filter):
    x_d = np.genfromtxt(
        X_train,
        dtype=np.float32,
        skip_header=1,
        delimiter=',',
        usecols=col_filter)
    t_d = np.genfromtxt(
        X_test,
        dtype=np.float32,
        skip_header=1,
        delimiter=',',
        usecols=col_filter)
    y_d = np.genfromtxt(
        Y_train,
        dtype=np.float32,
        skip_header=1,
        delimiter=',')
    x_d, t_d = normalize(x_d, t_d)

    return x_d, t_d, y_d
