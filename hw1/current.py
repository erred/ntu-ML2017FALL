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
        reader = csv.reader(inf)
        next(reader, None)
        for row in reader:
            if row[0] != lastid:
                arr.append(tarr)
                tarr = []
                lastid = row[0]
            tarr.append(row[:3] + [float(r) if r != 'NR' else 0 for r in row[3:]])
        arr.append(tarr)
    out = [d[3:] for d in arr[1]]
    for d in arr[2:]:
        for i in range(0,18):
            out[i].extend(d[i][3:])
    out2 = []
    for i in range(0,5760, 480):
        out2.append([r[i:i+480] for r in out])
    return out2

def savemodel(outputFile):
    print("saving: ", outputFile)
    with open(outputFile, 'w') as of:
        writer = csv.writer(of)
        writer.writerow([bias])
        writer.writerow(weights)

def loadModel(modelFile):
    global weights
    global bias
    with open(modelFile, 'r') as inf:
        reader = csv.reader(inf)
        row = next(reader)
        bias = float(row[0])
        row = next(reader)
        weights = [float(r) for r in row]

def predict(values):
    s = bias
    for h in range(0,9):
        i=9
        s += weights[h] * values[i][h]
        s += weights[h+9] * values[i][h] ** 2
    return s



def train(arr, outputFile):
    global bias
    global weights
    global bias_l
    global weights_l
    MAGIC = 9
    loss = 100000000000000
    counter = 0
    while(loss > 100):
        counter += 1
        loss = 0
        avgl = []
        for month in arr:
            weights_g = []
            bias_g = 0.0
            for j in range(0,18):
                weights_g.append(0.0)

            for i in range(0,480-9):
                section = [month[k][i:9+i] for k in range(0,18)]
                pred = predict(section)
                diff = month[MAGIC][9+i] - pred
                loss += diff ** 2
                bias_g -= diff * 2
                for h in range(0,9):
                    weights_g[h] -= 2 * diff * month[9][h+i]
                    weights_g[h+9] -= 2 * diff * month[9][h+i] ** 2
            bias_l += bias_g **2
            bias -= learn_rate * bias_g
            bias -= learn_rate /bias_l * bias_g
            for k in range(0,18):
                weights_l[k] += weights_g[k] ** 2
                weights[k] -= learn_rate / math.sqrt(weights_l[k]) * weights_g[k]
        print(loss, math.sqrt(loss/(471*12)))
        if counter % 100 == 0:
            savemodel("model/cp" + outputFile + "-" + str(int(loss)) + "-" + str(counter))
    savemodel("model/m" + outputFile + "-" + str(int(loss)) + "-" + str(counter))


def test(arr, outputFile):
    results = []
    for day in arr:
        section = [day[k][2:] for k in range(0,18)]
        pred = predict(section)
        pred = max(pred, 0)
        out = [day[0][0], pred]
        results.append(out)
        print(out)
        with open(outputFile, 'w') as of:
            writer = csv.writer(of)
            writer.writerow(["id", "value"])
            writer.writerows(results)


learn_rate = 0.001

weights = []
weights_l = []
for j in range(0,18):
    weights.append(0.0)
    weights_l.append(1.0)
bias = 0.0
bias_l = 1.0

def main(mode, inputFile, outputFile, modelFile):
    if mode == "train":
        loadModel(outputFile)
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
