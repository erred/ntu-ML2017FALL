import csv
import os
import sys

import numpy as np


def predict(xdata):
    out = 1.0 / (1.0 + np.exp(-1 * (np.matmul(xdata, weights) + bias)))
    return out

def train(xdata, ydata, iters, modelFile):
    global weights
    global weights_l
    global bias
    global bias_l
    counter = 0
    prevcost = 100
    cost = 90
    # for i in range(iters):
    while prevcost > cost:
        counter += 1
        if counter > 10:
            prevcost = cost
        pred = predict(xdata)

        loss = pred - ydata
        cost = - np.sum(np.multiply(ydata,np.log(pred)) + np.multiply((1-ydata),np.log(1-pred))) / len(loss)

        grad = np.dot(loss, xdata)
        weights_l += np.square(grad)
        weights -= np.dot(LEARNRATE, np.divide(grad, weights_l))
        bias_l += np.sum(loss ** 2)
        bias -= np.dot(LEARNRATE, np.divide(np.sum(loss), bias_l))
        print("epoch: ", counter, " cost: ", cost)
        if counter % 500 == 0:
            miss = np.abs(np.around(pred) - ydata)
            print("accuracy: ", 1.0 - np.sum(miss)/len(loss))
        if counter % 10000 == 0:
            saver(modelFile + "_" + str(counter))
    saver(modelFile + "_" + str(counter))


def saver(modelFile):
    with open(modelFile, 'w') as of:
        w = csv.writer(of)
        w.writerow([bias])
        w.writerow(weights.tolist())

def loader(modelFile):
    global bias
    global weights
    with open(modelFile, 'r') as of:
        r = csv.reader(of)
        row = next(r)
        bias = float(row[0])
        row = next(r)
        weights = np.array([float(x) for x in row])

def predOut(xdata, outputFile):
    pred = np.around(predict(xdata))
    with open(outputFile, 'w') as of:
        w = csv.writer(of)
        w.writerow(["id","label"])
        for i,x in enumerate(pred):
            w.writerow([i+1,int(x)])

LEARNRATE = 10
weights = np.zeros([106])
weights_l = np.zeros([106])
bias = 0.0
bias_l = 0.0

if __name__ == "__main__":
    if sys.argv[1] == "train":
        # train X_train, Y_train, modelFile
        loader(sys.argv[5])
        Xdata = np.genfromtxt(sys.argv[2], dtype=np.float32, skip_header=1, delimiter=',')
        Ydata = np.genfromtxt(sys.argv[3], dtype=np.float32, skip_header=1, delimiter=',')
        train(Xdata, Ydata, 300, sys.argv[4])
    else:
        # test modelFile X_test outfile
        modelFile = sys.argv[2]
        loader(modelFile)
        Xdata = np.genfromtxt(sys.argv[3], dtype=np.float32, skip_header=1, delimiter=',')
        outputFile = sys.argv[4]
        predOut(Xdata, outputFile)
