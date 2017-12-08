import csv
import operator
import os
import pickle
import sys

import gensim
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import (GRU, LSTM, RNN, Activation, Bidirectional, Conv1D,
                          Dense, Dropout, Embedding, Flatten, GRUCell, Input,
                          LSTMCell, Masking, StackedRNNCells, TimeDistributed)
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.regularizers import l1, l1_l2, l2
from keras.utils import plot_model, to_categorical

EMBEDDIMS = 32
VOCABSIZE = 248027
BATCHSIZE = 1024
EPOCHS = 1
# MODELDIR = 'model/' + os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELFILE ='model'

# tb = TensorBoard(log_dir=MODELDIR, histogram_freq=0, write_graph=True, write_grads=True)
# sv = ModelCheckpoint(MODELFILE, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
# =========================================================

inputs = Input(shape=(None, 100))

# trunk = Embedding(VOCABSIZE, EMBEDDIMS)(inputs)
trunk = Conv1D(128, 6, padding='same', activation='relu')(inputs)
trunk = Dropout(0.5)(trunk)
trunk = Bidirectional(LSTM(512, kernel_regularizer=l2(), kernel_initializer='he_normal'))(trunk)

trunk = Dropout(0.5)(trunk)
trunk = Dense(1024, activation='relu')(trunk)
trunk = Dropout(0.5)(trunk)
output = Dense(2, activation='softmax')(trunk)

model = Model(inputs=inputs, outputs=output)
opt = RMSprop(0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# plot_model(model, to_file=MODELDIR[6:]+'.png')
if os.path.isfile(MODELFILE):
    model.load_weights(MODELFILE)
    print('loaded pretrained model: ', MODELFILE)

# =========================================================

mode = sys.argv[1]
if mode == 'train':
    trainLabeled = sys.argv[2]
    trainUnlabeled = sys.argv[3]

    labeled_texts = []
    labels = []
    with open(trainLabeled, 'r') as fo:
        for line in fo:
            l = line.split()
            labeled_texts.append(' '.join(l[2:]))
            labels.append(int(l[0]))
    unlabeled_texts = []
    with open(trainUnlabeled, 'r') as fo:
        for line in fo:
            unlabeled_texts.append(line)

    if os.path.isfile('word2vec'):
        w2v = gensim.models.Word2Vec.load('word2vec')
        print('loaded pretrained word2vec')
    else:
        print('training new word2vec')
        sentences = [item for sublist in [labeled_texts, unlabeled_texts] for item in sublist]
        sentences = [text_to_word_sequence(line) for line in sentences]
        w2v = gensim.models.Word2Vec(sentences, min_count=1, iter=25)
        # if not os.path.exists(MODELDIR):
        #     os.makedirs(MODELDIR)
        w2v.save('word2vec')

    wv = w2v.wv
    del w2v

    seq = []
    for line in labeled_texts:
        ws = text_to_word_sequence(line)
        seq.append([wv[w] for w in ws])

    seq = pad_sequences(seq)
    lab = to_categorical(labels, 2)

    # model.fit(seq, lab, epochs=EPOCHS, batch_size=BATCHSIZE, validation_split=0.05, shuffle=True, callbacks=[tb, sv])

    labs = np.array_split(lab, 4)
    seqs = np.array_split(seq, 4)
    for k in range(4):
        for i in range(4):
            d = []
            l = []
            for j in range(4):
                if j != i:
                    d.append(seqs[j])
                    l.append(labs[j])
            model.fit(np.concatenate(d), np.concatenate(l), epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(seqs[i], labs[i]), shuffle=True)
    model.save_weights(MODELFILE)

else:
    inputFile = sys.argv[2]
    outputFile = sys.argv[3]

    # with open(MODELDIR+'/tokenizer.pkl', 'rb') as f:
    #     tokenizer = pickle.load(f)
    texts = []
    with open(inputFile, 'r') as fo:
        skip = next(fo)
        for line in fo:
            line = ','.join(line.split(',')[1:])
            texts.append(line)

    w2v = gensim.models.Word2Vec.load('word2vec')
    wv = w2v.wv
    del w2v

    seq = []
    for line in texts:
        ws = text_to_word_sequence(line)
        seq.append([wv[w] for w in ws])
    seq = pad_sequences(seq)

    predictions = model.predict(seq, batch_size=BATCHSIZE)
    print([p for p in predictions])
    predictions = np.argmax(predictions, axis=1)
    output = [[i, int(x)] for i, x in enumerate(predictions)]

    with open(outputFile, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['id','label'])
        writer.writerows(output)
