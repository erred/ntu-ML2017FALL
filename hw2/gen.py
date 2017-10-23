import csv
import sys

import numpy as np

import iofn

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

weights = np.zeros(FEATURES)
bias = np.zeros(1)

def saver(modelFile):
    with open(modelFile, 'w') as of:
        w = csv.writer(of)
        w.writerow(weights)
        w.writerow(bias)

def loader(modelFile):
    with open(modelFile, 'r') as of:
        r = csv.reader(of)
        row = next(r)
        weights = np.array([float(x) for x in row])
        row = next(r)
        bias = float(row[0])
        return weights, bias

def predict(arr):
    p = np.matmul(arr, weights) + bias
    out = 1 - iofn.activate(p)
    return out

def train(x, y, FEATURES=FEATURES):
    global weights
    global bias

    N1 = 0
    N2 = 0
    mu1 = np.zeros(FEATURES)
    mu2 = np.zeros(FEATURES)
    for i in range(len(y)):
        if y[i] == 1:
            N1 +=1
            mu1 += x[i]
        else:
            N2 += 1
            mu2 += x[i]
    mu1 /= N1
    mu2 /= N2

    sig1 = np.zeros((FEATURES, FEATURES))
    sig2 = np.zeros((FEATURES, FEATURES))
    for i in range(len(y)):
        if y[i] == 1:
            sig1 += np.dot(np.transpose([x[i] - mu1]), [(x[i] - mu1)])
        else:
            sig2 += np.dot(np.transpose([x[i] - mu2]), [(x[i] - mu2)])
    sig1 /= N1
    sig2 /= N2
    shared_sigma = ((float(N1) / len(y)) * sig1
                    + (float(N2) / len(y)) * sig2)
    sig_inv = np.linalg.inv(shared_sigma)
    weights = np.dot((mu1-mu2), sig_inv)
    bias = ((-0.5) * np.dot(np.dot([mu1], sig_inv), mu1)
            + (0.5) * np.dot(np.dot([mu2], sig_inv), mu2)
            + np.log(float(N1)/N2))

def test(x):
    return np.around(predict(x))

if __name__ == "__main__":
    mode = sys.argv[1]
    X_train = sys.argv[2]
    X_test = sys.argv[3]
    Y_train = sys.argv[4]
    x, t, y = iofn.readData(X_train, X_test, Y_train, col_filter)
    modelFile = sys.argv[5]

    if mode == "train":
        train(x, y)
        saver(modelFile)
    elif mode == "eval":
        weights, bias = loader(modelFile)
        output = test(x)
        print(100 * float(sum([int(y[i] == output[i]) for i in range(len(y))])) / len(y))
    else:
        outputFile = sys.argv[6]
        weights, bias = loader(modelFile)
        output = test(t)
        output = [[i+1,int(x)] for i,x in enumerate(output)]
        iofn.saveOutput(output, outputFile)
