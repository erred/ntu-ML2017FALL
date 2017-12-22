import csv
import os
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import (Add, BatchNormalization, Concatenate, Dense, Dot,
                          Dropout, Embedding, Flatten, Input, Multiply)
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.utils import to_categorical

MOVIES = 3952
USERS = 6040
OCCUPATIONS = 21
CATEGORIES = 36
AGES = 57

BATCHSIZE = 1024
EPOCHS = 100
MODELDIR = 'model/' + os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELFILE = MODELDIR + '/model'

tb = TensorBoard(log_dir=MODELDIR, embeddings_freq=1)
sv = ModelCheckpoint(MODELFILE, save_best_only=True, save_weights_only=True)
es = EarlyStopping(patience=3)

# =========================== Model

mi = Input([1])
ui = Input([1])
ag = Input([AGES])
oc = Input([OCCUPATIONS])
ca = Input([CATEGORIES])

m = Embedding(MOVIES, 16)(mi)
m = Flatten()(m)

u = Embedding(USERS, 16)(ui)
u = Flatten()(u)

bu = Concatenate()([ag, oc])
bu = Dense(1)(bu)

bm = Dense(1)(ca)

t = Dot(-1)([m, u])
t = Add()([t, bu, bm])

model = Model(inputs=[mi, ui, ag, oc, ca], outputs=t)
model.summary()
opt = 'adam'
model.compile(optimizer=opt, loss='mean_squared_error')

if os.path.isfile(MODELFILE):
    model.load_weights(MODELFILE)
    print('loaded pretrained model: ', MODELFILE)



def train():
    trainFile = 'data/train.csv'
    movieFile = 'data/movies.csv'
    userFile = 'data/users.csv'

    movies, categories, users = readFiles(movieFile, userFile)
    listing, lab = readListing(trainFile, True)
    use, mov, age, occ, cat = splitData(movies, users, listing)

    age = to_categorical(age, AGES)
    occ = to_categorical(occ, OCCUPATIONS)

    su = np.array_split(use, 10)
    sm = np.array_split(mov, 10)
    sa = np.array_split(age, 10)
    so = np.array_split(occ, 10)
    sc = np.array_split(cat, 10)
    sl = np.array_split(lab, 10)

    for i in range(EPOCHS):
        for j in range(10):
            bm, bu, ba, bo, bc, bl = [], [], [], [], [], []
            for k in range(10):
                if k != j:
                    bm.append(sm[k])
                    bu.append(su[k])
                    ba.append(sa[k])
                    bo.append(so[k])
                    bc.append(sc[k])
                    bl.append(sl[k])
            X = [np.concatenate(bm), np.concatenate(bu), np.concatenate(ba), np.concatenate(bo), np.concatenate(bc)]
            Y = np.concatenate(bl)
            VX = [sm[j], su[j], sa[j], so[j], sc[j]]
            VY = sl[j]
            model.fit(X, Y, batch_size=BATCHSIZE, epochs=1, validation_data=(VX, VY), shuffle=True, callbacks=[tb, sv])

    # model.fit([mov, use, age, occ, cat], lab, batch_size=BATCHSIZE, epochs=EPOCHS, validation_split=0.1, shuffle=True, callbacks=[tb, sv, es])


def test():
    testFile = sys.argv[2]
    movieFile = sys.argv[3]
    userFile = sys.argv[4]
    outputFile = sys.argv[5]

    movies, categories, users = readFiles(movieFile, userFile)
    listing = readListing(testFile)
    use, mov, age, occ, cat = splitData(movies, users, listing)

    age = to_categorical(age, AGES)
    occ = to_categorical(occ, OCCUPATIONS)

    predictions = model.predict([mov, use, age, occ, cat], batch_size=BATCHSIZE)
    output = [[i+1, np.minimum(x[0], 5)] for i, x in enumerate(predictions)]

    with open(outputFile, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['TestDataID','Rating'])
        writer.writerows(output)


def readListing(filename, labels=False):
    listing = []
    lab = []
    with open(filename) as f:
        next(f)
        for line in f:
            # id, userid, movieid
            l = line.replace('\\n', '').split(',')
            listing.append([int(i) for i in l[:3]])
            if labels:
                lab.append(l[3])
    lab = np.array(lab)
    if labels:
        return listing, lab
    return listing

def splitData(movies, users, listing):
    use = []
    mov = []
    age = []
    occ = []
    cat = []
    for line in listing:
        c = [0] * CATEGORIES
        for ca in movies[line[2]]['catsid']:
            c[ca] = 1

        cat.append(c)
        use.append(line[1])
        mov.append(line[2])
        age.append(users[line[1]]['age'])
        occ.append(users[line[1]]['occ'])

    use = np.array(use)
    mov = np.array(mov)
    age = np.array(age)
    occ = np.array(occ)
    cat = np.array(cat)
    return use, mov, age, occ, cat

def readFiles(movieFile, userFile):
    users = {}
    with open(userFile) as f:
        next(f)
        for line in f:
            # id, gender, age, occupation, zip
            l = line.replace('\\n', '').split('::')
            g = 1
            if l[1] == "F":
                g = 0

            users[int(l[0])] = {'gender': g, 'age': int(l[2]), 'occ': int(l[3]), 'zip': l[4]}

    categories = ['']
    movies = {}
    with open(movieFile, encoding='latin-1') as f:
        next(f)
        for line in f:
            # id, title, cat|egor|ies
            l = line.replace('\\n', '').split('::')
            cats = l[2].split('|')
            for c in cats:
                if c not in categories:
                    categories.append(c)

            movies[int(l[0])] = {'title': l[1], 'cats': cats, 'catsid':[categories.index(c) for c in cats]}
    return movies, categories, users

if __name__ == "__main__":
    MODE = sys.argv[1]
    if MODE == 'train':
        train()
    else:
        test()
