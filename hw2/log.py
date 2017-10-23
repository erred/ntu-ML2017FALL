import csv
import sys

import numpy as np

import iofn

col_filter = []
# col_filter.append(list(range(0, 1)))  # age
# col_filter.append(list(range(1, 2)))  # fnlwgt
# col_filter.append(list(range(2, 3)))  # sex
# col_filter.append(list(range(3, 5)))  # capital gain/loss
# col_filter.append(list(range(5, 6)))  # hours_per_week
# col_filter.append(list(range(6, 15)))  # employer
# col_filter.append(list(range(15, 22)))  # edu_num
# col_filter.append(list(range(22, 31)))  # edu
# col_filter.append(list(range(31, 38)))  # maritial
# col_filter.append(list(range(38,53))) # occupation
col_filter.append(list(range(53, 59)))  # relationship
# col_filter.append(list(range(59,64))) # race
col_filter.append(list(range(64, 106)))  # country
col_filter = sum(col_filter, [])

FEATURES = len(col_filter)
EPOCHS = 1000
LEARNRATE = 1

weights = np.zeros([FEATURES])
weights_l = np.zeros([FEATURES])
# weights2 = np.zeros([FEATURES])
# weights2_l = np.zeros([FEATURES])
bias = 0.0
bias_l = 0.0

def saver(modelFile):
    with open(modelFile, 'w') as of:
        w = csv.writer(of)
        w.writerow([bias])
        w.writerow(weights.tolist())
        # w.writerow(weights2.tolist())

def loader(modelFile):
    with open(modelFile, 'r') as of:
        r = csv.reader(of)
        row = next(r)
        bias = float(row[0])
        row = next(r)
        weights = np.array([float(x) for x in row])
        # row = next(r)
        # weights2 = np.array([float(x) for x in row])
    # return weights, weights2, bias
    return weights, bias

def predict(arr, weights, bias):
    # p = np.matmul(arr, weights) + np.matmul(np.square(arr), weights2) + bias
    p = np.matmul(arr, weights) + bias
    p = iofn.activate(p)
    return p

def train(x, y, epochs, weights=weights, weights_l=weights_l, bias=bias, bias_l=bias_l):

    for idx in range(epochs):
        pred = predict(x, weights, bias)
        loss = y - pred
        xentropy  = -1 * np.sum(np.multiply(y,np.log(pred)) + np.multiply((1-y),np.log(1-pred))) / len(loss)

        grad = np.sum(-1 * np.dot(loss, x))
        weights_l += np.square(grad)
        weights -= np.dot(LEARNRATE, np.divide(grad, weights_l))

        # grad2 = np.dot(loss, np.square(x))
        # weights2_l += np.square(grad2)
        # weights2 += np.dot(LEARNRATE, np.divide(grad2, weights2_l))

        bias_grad = np.sum(-1 * loss)
        bias_l += np.square(bias_grad)
        bias -= np.dot(LEARNRATE, np.divide(bias_grad, bias_l))

        # print(sum(abs(loss)))
        # print(xentropy)
    return weights, weights_l, bias, bias_l

def test(x, weights, bias):
    return np.around(predict(x, weights, bias))


if __name__ == "__main__":
    mode = sys.argv[1]
    X_train = sys.argv[2]
    X_test = sys.argv[3]
    Y_train = sys.argv[4]
    x, t, y = iofn.readData(X_train, X_test, Y_train, col_filter)
    modelFile = sys.argv[5]

    if mode == "train":
        weights, weights_l, bias, bias_l = train(x, y, EPOCHS, weights, weights_l, bias, bias_l)
        saver(modelFile)
        output = test(x, weights, bias)
        print(100 * float(sum([int(y[i] == output[i]) for i in range(len(y))])) / len(y))
    else:
        outputFile = sys.argv[6]
        # weights, weights2, bias = loader(modelFile)
        weights, bias = loader(modelFile)
        output = test(t, weights, bias)
        output = [[i+1,int(x)] for i,x in enumerate(output)]
        iofn.saveOutput(output, outputFile)
