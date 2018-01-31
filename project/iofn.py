import csv
import pickle

import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNKONW_TOKEN = 3

KEEP = -4000


def trainFn(batch_size=128, epochs=20):
    dataFile = 'data/train.data'
    labelFile = 'data/train.caption'

    with open(dataFile, 'rb') as f:
        sound = pickle.load(f)

    with open('w2i.pkl', 'rb') as f:
        w2i = pickle.load(f)
    counter = len(w2i)
    texts = []
    with open(labelFile) as f:
        # texts = f.readlines()
        re = csv.reader(f, delimiter=' ')
        for r in re:
            tex = [1]
            for c in r:
                if c not in w2i:
                    w2i[c] = counter
                    counter += 1
                tex.append(w2i[c])
            tex.append(2)
            texts.append(tex)

    with open('w2i.pkl', 'wb') as f:
        pickle.dump(w2i, f)
    i2w = {0: ' ', 1: ' ', 2: ' ', 3: ' '}
    for k, v in w2i.items():
        i2w[v] = k
    with open('i2w.pkl', 'wb') as f:
        pickle.dump(i2w, f)

    lens = np.array([len(t) for t in texts])
    # maxlen = max(lens)
    maxlen = 16
    for i in range(len(texts)):
        texts[i] = texts[i] + [0] * (maxlen - lens[i])

    seqlens = np.array([x.shape[0] for x in sound])
    maxlen = 246
    sound = pad_sequences(np.array(sound), maxlen)
    texts = np.array(texts)

    trfn = tf.estimator.inputs.numpy_input_fn(
        x={'x': sound[:KEEP],
           'lens': lens[:KEEP],
           'seqlens': seqlens[:KEEP]},
        y=texts[:KEEP],
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)
    evfn = tf.estimator.inputs.numpy_input_fn(
        x={'x': sound[KEEP:],
           'lens': lens[KEEP:],
           'seqlens': seqlens[KEEP:]},
        y=texts[KEEP:],
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=False)
    return trfn, evfn


def testFn(batch_size=128):
    dataFile = 'data/test.data'
    choiceFile = 'data/test.csv'

    with open(dataFile, 'rb') as f:
        sound = pickle.load(f)

    with open('w2i.pkl', 'rb') as f:
        w2i = pickle.load(f)

    texts = []
    with open(choiceFile) as f:
        re = csv.reader(f)
        for r in re:
            r = [w.split() for w in r]
            for i in range(4):
                r[i] = [1] + [w2i.get(x, 3) for x in r[i]] + [2]
                r[i] = r[i] + [0] * (16 - len(r[i]))
            texts.append(r)

    lens = []
    for i in range(len(texts)):
        ls = []
        for j in range(len(texts[i])):
            ls.append(len(texts[i][j]))
            texts[i][j] = texts[i][j] + [0] * (16 - len(texts[i][j]))
        lens.append(ls)

    texts = np.array(texts)

    seqlens = np.array([x.shape[0] for x in sound])
    sound = pad_sequences(np.array(sound))
    lens = np.array(lens)

    fn = tf.estimator.inputs.numpy_input_fn(
        # x={'x':sound, 'seqlens': seqlens},
        x={'x': sound,
           'seqlens': seqlens,
           'choices': texts,
           'lens': lens},
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)
    return fn, texts


def ModTrainFn(batch_size=128, epochs=20):
    dataFile = 'data/train.data'
    labelFile = 'data/train.caption'

    with open(dataFile, 'rb') as f:
        sound = pickle.load(f)

    cf = {}
    with open(labelFile) as f:
        re = csv.reader(f, delimiter=' ')
        for r in re:
            for c in r:
                if c not in cf:
                    cf[c] = 0
                cf[c] += 1
    with open('cf.pkl', 'wb') as f:
        pickle.dump(cf, f)

    with open('w2i.pkl', 'rb') as f:
        w2i = pickle.load(f)
    counter = len(w2i)
    texts = []
    with open(labelFile) as f:
        re = csv.reader(f, delimiter=' ')
        for r in re:
            tex = [1]
            for c in r:
                if c not in w2i:
                    w2i[c] = counter
                    counter += 1
                if cf[c] < 20:
                    tex.append(3)
                else:
                    tex.append(w2i[c])
            tex.append(2)
            texts.append(tex)

    with open('w2i.pkl', 'wb') as f:
        pickle.dump(w2i, f)
    i2w = {0: ' ', 1: ' ', 2: ' ', 3: ' '}
    for k, v in w2i.items():
        i2w[v] = k
    with open('i2w.pkl', 'wb') as f:
        pickle.dump(i2w, f)

    lens = np.array([len(t) for t in texts])
    maxlen = max(lens)
    print('records: ', len(lens))
    print('max ans len: ', maxlen)
    for i in range(len(texts)):
        texts[i] = texts[i] + [0] * (maxlen - lens[i])

    seqlens = np.array([x.shape[0] for x in sound])
    print('max seq len: ', max(seqlens))
    sound = pad_sequences(np.array(sound))
    texts = np.array(texts)

    trfn = tf.estimator.inputs.numpy_input_fn(
        x={'x': sound[:KEEP],
           'lens': lens[:KEEP],
           'seqlens': seqlens[:KEEP]},
        y=texts[:KEEP],
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)
    evfn = tf.estimator.inputs.numpy_input_fn(
        x={'x': sound[KEEP:],
           'lens': lens[KEEP:],
           'seqlens': seqlens[KEEP:]},
        y=texts[KEEP:],
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=False)
    return trfn, evfn


def ModTestFn(batch_size=128):
    dataFile = 'data/test.data'
    choiceFile = 'data/test.csv'

    with open(dataFile, 'rb') as f:
        sound = pickle.load(f)

    with open('w2i.pkl', 'rb') as f:
        w2i = pickle.load(f)

    with open('cf.pkl', 'rb') as f:
        cf = pickle.load(f)

    texts = []
    with open(choiceFile) as f:
        re = csv.reader(f)
        for r in re:
            r = [w.split() for w in r]
            for i in range(4):
                r[i] = [1] + [
                    w2i.get(x, 3) if cf.get(x, 0) > 19 else 3 for x in r[i]
                ] + [2]
                r[i] = r[i] + [0] * (16 - len(r[i]))
            texts.append(r)

    lens = []
    for i in range(len(texts)):
        ls = []
        for j in range(len(texts[i])):
            ls.append(len(texts[i][j]))
            texts[i][j] = texts[i][j] + [0] * (16 - len(texts[i][j]))
        lens.append(ls)

    texts = np.array(texts)

    seqlens = np.array([x.shape[0] for x in sound])
    sound = pad_sequences(np.array(sound))
    lens = np.array(lens)

    fn = tf.estimator.inputs.numpy_input_fn(
        # x={'x':sound, 'seqlens': seqlens},
        x={'x': sound,
           'seqlens': seqlens,
           'choices': texts,
           'lens': lens},
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)
    return fn, texts


if __name__ == '__main__':
    trainFn()
