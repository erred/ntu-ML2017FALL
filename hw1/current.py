import csv
import sys
import math
import random
# import numpy as np
# import scipy as sp

def readTestData(inputFile):
    arr = []
    lastid = ""
    tarr = []
    with open(inputFile, 'r') as inf:
        reader = csv.reader(inf)
        for row in reader:
            #id, param, 9
            if row[0] != lastid:
                arr.append(tarr)
                tarr = []
                lastid = row[0]
            tarr.append(row[:2] + [float(r) if r != 'NR' else 0 for r in row[2:]])
        arr.append(tarr)
    return arr[1:]

def readTrainData(inputFile):
    arr = []
    lastid = ""
    tarr = []
    with open(inputFile, 'r', encoding='latin1') as inf:
        #id, ???, param, 24
        reader = csv.reader(inf)
        next(reader, None)
        for row in reader:
            if row[0] != lastid:
                arr.append(tarr)
                tarr = []
                lastid = row[0]
            tarr.append(row[:3] + [float(r) if r != 'NR' else 0 for r in row[3:]])
        arr.append(tarr)
    return arr[1:]

def savemodel(outputFile):
    print("saving: ", outputFile)
    with open(outputFile, 'w') as of:
        writer = csv.writer(of)
        writer.writerow([bias])
        writer.writerows(weights)

def loadModel(modelFile):
    global weights
    global bias
    with open(modelFile, 'r') as inf:
        reader = csv.reader(inf)
        row = next(reader)
        bias = float(row[0])
        counter = 0
        for row in reader:
            weights[counter] = [float(r) for r in row]
            counter += 1

def predict(values):
    s = bias
    for h in range(0,9):
        for i in range(0,18):
            s += weights[h][i] * values[i][h]
    return s



def train(arr, outputFile):
    global bias
    global weights
    global bias_l
    global weights_l
    MAGIC = 9
    loss = 100000000000000
    counter = 0
    while(loss > 100000):
        counter += 1
        loss = 0
        for day in arr:
            weights_g = []
            bias_g = 0.0
            for j in range(0,9):
                w = []
                for i in range(0,18):
                    w.append(float(0))
                weights_g.append(w)
            for i in range(0,24-9):
                section = [day[k][3+i:12+i] for k in range(0,18)]
                pred = predict(section)
                diff = day[MAGIC][12+i] - pred
                loss += diff ** 2
                bias_g -= diff * 2
                for h in range(0,9):
                    for j in range(0,18):
                        weights_g[h][j] -= 2 * diff * day[j][h+i+3]
            bias_l += bias_g **2
            bias -= learn_rate / math.sqrt(bias_l) * bias_g
            for k in range(0,9):
                for j in range(0,18):
                    weights_l[k][j] += weights_g[k][j] ** 2
                    weights[k][j] -= learn_rate / math.sqrt(weights_l[k][j]) * weights_g[k][j]
        print(loss)
        if counter % 100 == 0:
            savemodel("model/cp" + outputFile + "-" + str(int(loss)) + "-" + str(counter))
    savemodel("model/m" + outputFile + "-" + str(int(loss)) + "-" + str(counter))


def test(arr, outputFile):
    results = []
    for day in arr:
        section = [day[k][2:] for k in range(0,18)]
        pred = predict(section)
        pred = max(round(pred), 0)
        out = [day[0][0], pred]
        results.append(out)
        print(out)
        with open(outputFile, 'w') as of:
            writer = csv.writer(of)
            writer.writerow(["id", "value"])
            writer.writerows(results)



random.seed()

learn_rate = 10

weights = []
weights_l = []
for j in range(0,9):
    w = []
    w_l = []
    for i in range(0,18):
        w.append(float(0))
        w_l.append(float(1))
    weights.append(w)
    weights_l.append(w_l)
bias = 0.0
bias_l = 1.0

def main(mode, inputFile, outputFile, modelFile):
    if mode == "train":
        # loadModel(outputFile)
        data = readTrainData(inputFile)
        train(data, modelFile)
        savemodel(outputFile)
    else:
        loadModel(modelFile)
        data = readTestData(inputFile)
        test(data, outputFile)

if __name__ == "__main__":
    mode = sys.argv[1]
    inputFile = sys.argv[2]
    outputFile = sys.argv[3]
    modelFile = sys.argv[4]
    main(mode, inputFile, outputFile, modelFile)
