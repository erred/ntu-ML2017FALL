import csv

import numpy as np


def saveOutput(arr, filename):
    with open(filename, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['id','label'])
        writer.writerows(arr)

def readTrain(filepath):
    with open(filepath, 'rb') as fo:
        input = np.genfromtxt(
            (f.replace(b',',b' ') for f in fo),
            skip_header=1)
        label = input[:,0]
        data = np.reshape(input[:,1:], (-1, 48, 48))
        return label.astype(int), np.divide(data - 128, 255)

def readTest(filepath):
    with open(filepath, 'rb') as fo:
        input = np.genfromtxt(
            (f.replace(b',',b' ') for f in fo),
            skip_header=1)
        label = input[:,0]
        data = np.reshape(input[:,1:], (-1, 48, 48))
        return label.astype(int), np.divide(data - 128, 255)

def readTrainNoSqueeze(filepath):
    with open(filepath, 'rb') as fo:
        input = np.genfromtxt(
            (f.replace(b',',b' ') for f in fo),
            skip_header=1)
        label = input[:,0]
        data = np.reshape(input[:,1:], (-1, 48, 48))
        return label.astype(int), data - 128

def readTestNoSqueeze(filepath):
    with open(filepath, 'rb') as fo:
        input = np.genfromtxt(
            (f.replace(b',',b' ') for f in fo),
            skip_header=1)
        label = input[:,0]
        data = np.reshape(input[:,1:], (-1, 48, 48))
        return label.astype(int), data - 128

def readTrainRaw(filepath):
    with open(filepath, 'rb') as fo:
        input = np.genfromtxt(
            (f.replace(b',',b' ') for f in fo),
            skip_header=1)
        label = input[:,0]
        data = np.reshape(input[:,1:], (-1, 48, 48))
        return label.astype(int), data

def readTestRaw(filepath):
    with open(filepath, 'rb') as fo:
        input = np.genfromtxt(
            (f.replace(b',',b' ') for f in fo),
            skip_header=1)
        label = input[:,0]
        data = np.reshape(input[:,1:], (-1, 48, 48))
        return label.astype(int), data
