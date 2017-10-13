import csv
import sys
import math
import random


def readTestData(inputFile):
    arr = []
    lastid = ""
    tarr = []
    with open(inputFile, 'r') as inf:
        reader = csv.reader(inf)
        for row in reader:
            #id, param, 9
            if row[0] == lastid:
                tarr.append(row[:2] + [float(r) if r !=
                                       'NR' else 0 for r in row[2:]])
            else:
                arr.append(tarr)
                tarr = []
                lastid = row[0]
                tarr.append(row[:2] + [float(r) if r !=
                                       'NR' else 0 for r in row[2:]])
        arr.append(tarr)
    return arr[1:]


def readTrainData(inputFile):
    arr = []
    lastid = ""
    tarr = []
    with open(inputFile, 'r', encoding='latin1') as inf:
        # id, ???, param, 24
        reader = csv.reader(inf)
        next(reader, None)
        for row in reader:
            if row[0] == lastid:
                tarr.append(row[:3] + [float(r) if r !=
                                       'NR' else 0 for r in row[3:]])
            else:
                arr.append(tarr)
                tarr = []
                lastid = row[0]
                tarr.append(row[:3] + [float(r) if r !=
                                       'NR' else 0 for r in row[3:]])
        arr.append(tarr)
    return arr[1:]


def predict(values):
    s = bias
    for h in TIMERANGE:
        for i in FRANGE:
            s += weights[h][i] * values[i][h]
    return s


def train(arr):
    global bias
    global weights
    global bias_l
    global weights_l
    MAGIC = 9
    loss = 100000000000000
    counter = 0
    while(counter < 4000):
        counter += 1
        loss = 0
        for day in arr:
            weights_g = []
            bias_g = 0.0
            for j in range(0, 9):
                w = []
                for i in range(0, 18):
                    w.append(float(0))
                weights_g.append(w)
            for i in range(0, 24 - 9):
                section = [day[k][3 + i:12 + i] for k in range(0, 18)]
                pred = predict(section)
                diff = day[MAGIC][12 + i] - pred
                loss += diff ** 2
                bias_g -= diff * 2 + LAMBDA * 2 * bias
                for h in TIMERANGE:
                    for j in FRANGE:
                        weights_g[h][j] -= 2 * diff * \
                            day[j][h + i + 3] + LAMBDA * 2 * weights[h][j]
            bias_l += bias_g ** 2
            bias -= learn_rate / math.sqrt(bias_l) * bias_g
            for k in TIMERANGE:
                for j in FRANGE:
                    weights_l[k][j] += weights_g[k][j] ** 2
                    weights[k][j] -= learn_rate / \
                        math.sqrt(weights_l[k][j]) * weights_g[k][j]
        print(loss)


def test(arr, outputFile):
    results = []
    for day in arr:
        section = [day[k][2:] for k in range(0, 18)]
        pred = predict(section)
        pred = max(round(pred), 0)
        out = [day[0][0], pred]
        results.append(out)
        print(out)
        with open(outputFile, 'w') as of:
            writer = csv.writer(of)
            writer.writerow(["id", "value"])
            writer.writerows(results)


weights = []
weights_l = []
bias = 0.0
bias_l = 0.0
learn_rate = 10
LAMBDA = 0.0


def load(model):
    global bias
    global weights
    with open(model, 'r') as fo:
        r = csv.reader(fo)
        row = next(r)
        bias = float(row[0])
        weights = []
        counter = 1
        for row in r:
            weights.append([float(x) for x in row])
            counter += 1


def reset():
    global weights
    global weights_l
    global bias
    global bias_l
    weights = []
    for j in range(0, 9):
        w = []
        for i in range(0, 18):
            w.append(float(0))
        weights.append(w)
    bias = 0.0
    weights_l = []
    for j in range(0, 9):
        w = []
        for i in range(0, 18):
            w.append(float(1))
        weights_l.append(w)
    bias_l = 1.0


TIMERANGE = []
FRANGE = []


def main(model, testIn, out):
    global TIMERANGE
    global FRANGE
    global LAMBDA
    # trdata = readTrainData(trainIn)
    tedata = readTestData(testIn)

    reset()
    TIMERANGE = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    FRANGE = [9]
    # train(trdata)
    # with open('best.model', 'w') as fo:
    #     w = csv.writer(fo)
    #     w.writerow([bias])
    #     w.writerows(weights)
    load(model)
    test(tedata, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
