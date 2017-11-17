import os
import sys

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Concatenate, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Input,
                          MaxPooling2D)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import iofn

# ========== Hyper
BATCHSIZE = 128
STARTEPOCH = 0
EPOCHS = 160
MODELDIR = 'model/cnn53/'
MODELFILE = MODELDIR + 'model'

# ========== Custom
def C2N(x, units, kernel, strides=1):
    x = Conv2D(units, kernel, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def blockA(x):
    branch0 = C2N(x, 96, 1)

    branch1 = C2N(x, 32, 1)
    branch1 = C2N(branch1, 96, 3)

    branch2 = C2N(x, 64, 1)
    branch2 = C2N(branch2, 96, 3)
    branch2 = C2N(branch2, 96, 3)

    branch3 = AveragePooling2D(3, strides=1, padding='same')(x)
    branch3 = C2N(branch3, 96, 1)

    return Concatenate()([branch0, branch1, branch2, branch3])

def blockB(x):
    branch0 = C2N(x, 384, 1)

    branch1 = C2N(x, 192, 1)
    branch1 = C2N(branch1, 224, (1, 4))
    branch1 = C2N(branch1, 256, (4, 1))

    branch2 = C2N(x, 192, 1)
    branch2 = C2N(branch2, 192, (1, 4))
    branch2 = C2N(branch2, 224, (4, 1))
    branch2 = C2N(branch2, 224, (1, 4))
    branch2 = C2N(branch2, 256, (4, 1))

    branch3 = AveragePooling2D(3, strides=1, padding='same')(x)
    branch3 = C2N(branch3, 128, 1)

    return Concatenate()([branch0, branch1, branch2, branch3])

def blockC(x):
    branch0 = C2N(x, 256, 1)

    branch1 = C2N(x, 384, 1)
    branch1a = C2N(branch1, 256, (1, 2))
    branch1b = C2N(branch1, 256, (2, 1))
    branch1 = Concatenate()([branch1a, branch1b])

    branch2 = C2N(x, 384, 1)
    branch2 = C2N(branch2, 448, (1, 2))
    branch2 = C2N(branch2, 512, (2, 1))
    branch2a = C2N(branch2, 256, (1, 4))
    branch2b = C2N(branch2, 256, (4, 1))
    branch2 = Concatenate()([branch2a, branch2b])

    branch3 = AveragePooling2D(3, strides=1, padding='same')(x)
    branch3 = C2N(branch3, 256, 1)

    return Concatenate()([branch0, branch1, branch2, branch3])

# ========== Model
if os.path.isfile(MODELFILE):
    model = load_model(MODELFILE)
else:
    inputs = Input(shape=(48,48,1))

    # simple cnn
    trunk = C2N(inputs, 32, 2)
    trunk = C2N(trunk, 32, 3, strides=2)
    trunk = C2N(trunk, 64, 1)

    # pod 1
    branch0 = MaxPooling2D(2, strides=1, padding='same')(trunk)
    branch1 = C2N(trunk, 96, 2)
    trunk = Concatenate()([branch0, branch1])

    # pod 2
    branch0 = C2N(trunk, 64, 1)
    branch0 = C2N(branch0, 96, 2)
    branch1 = C2N(trunk, 64, 1)
    branch1 = C2N(branch1, 64, (1, 4))
    branch1 = C2N(branch1, 64, (4, 1))
    branch1 = C2N(branch1, 96, 2)
    trunk = Concatenate()([branch0, branch1])

    # pod 3
    branch0 = C2N(trunk, 192, 3, 2)
    branch1 = MaxPooling2D(3, strides=2, padding='same')(trunk)
    trunk = Concatenate()([branch0, branch1])

    # block-A *4
    trunk = blockA(trunk)
    trunk = blockA(trunk)
    # trunk = blockA(trunk)
    # trunk = blockA(trunk)

    # block A reduce
    branch0 = C2N(trunk, 384, 3, 2)
    branch1 = C2N(trunk, 192, 1)
    branch1 = C2N(branch1, 224, 3)
    branch1 = C2N(branch1, 256, 3, 2)
    branch2 = MaxPooling2D(3, strides=2, padding='same')(trunk)
    trunk = Concatenate()([branch0, branch1, branch2])

    # block B *7
    trunk = blockB(trunk)
    trunk = blockB(trunk)
    trunk = blockB(trunk)
    # trunk = blockB(trunk)
    # trunk = blockB(trunk)
    # trunk = blockB(trunk)
    # trunk = blockB(trunk)

    # block B  reduce
    branch0 = C2N(trunk, 192, 1)
    branch0 = C2N(branch0, 192, 3, 2)
    branch1 = C2N(trunk, 256, 1)
    branch1 = C2N(branch1, 256, (1, 4))
    branch1 = C2N(branch1, 320, (4, 1))
    branch1 = C2N(branch1, 320, 3, 2)
    branch2 = MaxPooling2D(3, strides=2, padding='same')(trunk)
    trunk = Concatenate()([branch0, branch1, branch2])

    # block C *3
    trunk = blockC(trunk)
    # trunk = blockC(trunk)
    # trunk = blockC(trunk)

    # prediction head
    output = AveragePooling2D(2, padding='valid')(trunk)
    output = Dropout(0.9)(output)
    output = Flatten()(output)
    output = Dense(7, activation='softmax')(output)


    model = Model(inputs=inputs, outputs=output)

    opt = Adadelta()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()



# ========== Main
mode = sys.argv[1]
inputFile = sys.argv[2]
if mode == "train":
    labels, data = iofn.readTrainRaw(inputFile)
    data_ed = np.expand_dims(data, 3)
    labels_oh = to_categorical(labels, num_classes=7)
    tb = TensorBoard(log_dir=MODELDIR, histogram_freq=0, write_graph=True, write_grads=True)

    # data_ed = data_ed[:1000]
    # labels_oh = labels_oh[:1000]

    split = int(len(data_ed) * 0.9)
    data_t = data_ed[:split]
    labels_t = labels_oh[:split]
    data_l = data_ed[split:]
    labels_l = labels_oh[split:]

    idg = ImageDataGenerator(rotation_range=30, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5, shear_range=0.2, horizontal_flip=True)

    model.fit_generator(idg.flow(data_t, labels_t, batch_size=BATCHSIZE), steps_per_epoch=len(data_t)/BATCHSIZE, epochs=EPOCHS, callbacks=[tb], validation_data=(data_l, labels_l), initial_epoch=STARTEPOCH)

    # model.fit(data_ed, labels_oh, epochs=EPOCHS, batch_size=BATCHSIZE, validation_split=0.1, shuffle=True, callbacks=[tb])
    model.save(MODELFILE)

if mode == "eval":
    labels, data = iofn.readTrainRaw(inputFile)
    data_ed = np.expand_dims(data, 3)
    labels_oh = to_categorical(labels, num_classes=7)

    loss = model.evaluate(data_ed, labels_oh, batch_size=BATCHSIZE)
    print(sum(loss))

if mode == "test":
    outputFile = sys.argv[3]
    labels, data = iofn.readTestRaw(inputFile)
    data_ed = np.expand_dims(data, 3)

    predictions = model.predict(data_ed, batch_size=BATCHSIZE)
    predictions = np.argmax(predictions, axis=1)

    output = [[x[0], int(x[1])] for x in zip(labels, predictions)]
    iofn.saveOutput(output, outputFile)
