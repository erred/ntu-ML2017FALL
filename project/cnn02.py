import csv
import os
import sys
import pickle

import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

KEEP = -4000

BATCHSIZE = 256
EPOCHS = 200
EMBED = 32
LEARNING_RATE = 0.001
LOGSTEPS = 160
EVALSTEPS = 15
# =================== Constants
# RECORDS = 45036

VOCAB = 2389 + 4 + 1
MAX_SEQ_LEN = 256
MAX_OUTPUT_LEN = 16

PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNKOWN_TOKEN = 3

VERSION = os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELDIR = 'model/' + VERSION
tf.logging.set_verbosity(tf.logging.INFO)


def convblock(x,
              filters=128,
              kernel=8,
              strides=1,
              padding='same',
              activation=tf.nn.leaky_relu,
              bn=True,
              training=False):
    x = tf.layers.conv1d(
        x,
        filters=filters,
        kernel_size=kernel,
        strides=strides,
        padding=padding)
    if bn:
        x = tf.layers.batch_normalization(x, training=training)
    if activation is not None:
        x = activation(x)
    return x


def model_fn(features, labels, mode):

    # =================== Inputs
    lens = features['lens']

    training = False
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True

    if mode == tf.estimator.ModeKeys.PREDICT:
        labels = tf.cast(features['choices'], tf.int64)

    # =================== Model Starts Here
    x = tf.cast(features['x'], tf.float32)

    x = convblock(x, 64, 8, 1, training=training)
    x = convblock(x, 128, 8, 2, training=training)
    x = convblock(x, 32, 8, 1, training=training)

    x = convblock(x, 128, 8, 1, training=training)
    x = convblock(x, 256, 8, 2, training=training)
    x = convblock(x, 64, 8, 1, training=training)

    x = convblock(x, 256, 8, 1, training=training)
    x = convblock(x, 512, 8, 2, training=training)
    x = convblock(x, 128, 8, 1, training=training)

    x = convblock(x, 512, 8, 1, training=training)
    x = convblock(x, 1024, 8, 2, training=training)
    x = convblock(x, VOCAB, 8, 1, bn=False, activation=None, training=training)

    output = x
    # =================== Model Ends Here
    return modelSpec(mode, output, labels, lens)


# =================== Make TF run
def modelSpec(mode, output, labels, lens):
    if mode == tf.estimator.ModeKeys.PREDICT:
        mask = tf.cast(
            tf.sequence_mask(tf.reshape(lens, [-1]), MAX_OUTPUT_LEN),
            tf.float32)
        labels = tf.reshape(labels, [-1, MAX_OUTPUT_LEN])
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=tf.contrib.seq2seq.tile_batch(output, 4),
            targets=labels,
            weights=mask,
            average_across_batch=False)
        loss = tf.reshape(loss, [-1, 4])
        loss = tf.argmin(loss, 1)

        predictions = {'seq': tf.argmax(output, 2), 'choice': loss}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        mask = tf.cast(tf.sequence_mask(lens, MAX_OUTPUT_LEN), tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=output, targets=labels, weights=mask)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=LEARNING_RATE,
            optimizer=optimizer)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)


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
    for i in range(len(texts)):
        texts[i] = texts[i] + [0] * (MAX_OUTPUT_LEN - lens[i])

    seqlens = np.array([x.shape[0] for x in sound])
    sound = pad_sequences(np.array(sound), MAX_SEQ_LEN)
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
                r[i] = r[i] + [0] * (MAX_OUTPUT_LEN - len(r[i]))
            texts.append(r)

    lens = []
    for i in range(len(texts)):
        ls = []
        for j in range(len(texts[i])):
            ls.append(len(texts[i][j]))
        lens.append(ls)

    texts = np.array(texts)

    seqlens = np.array([x.shape[0] for x in sound])
    sound = pad_sequences(np.array(sound), MAX_SEQ_LEN)
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


if __name__ == "__main__":

    runConfig = tf.estimator.RunConfig()
    runConfig = runConfig.replace(
        log_step_count_steps=50,
        keep_checkpoint_max=2,
        save_checkpoints_steps=LOGSTEPS,
        save_summary_steps=LOGSTEPS)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=None, config=runConfig, model_dir=MODELDIR)

    # train_fn, eval_fn = trainFn(batch_size=BATCHSIZE, epochs=EPOCHS)
    # experiment = tf.contrib.learn.Experiment(
    #     estimator=estimator,
    #     train_input_fn=train_fn,
    #     eval_input_fn=eval_fn,
    #     eval_steps=EVALSTEPS,
    #     eval_delay_secs=1,
    #     min_eval_frequency=1)

    # experiment.train_and_evaluate()

    test_fn, texts = testFn(batch_size=BATCHSIZE)
    preds = estimator.predict(input_fn=test_fn)
    preds = [[p['seq'], p['choice']] for p in preds]
    # print([p[1] for p in preds])
    with open('out/' + VERSION, 'w') as f:
        w = csv.writer(f)
        w.writerow(['id', 'answer'])
        w.writerows([[i + 1, p[1]] for i, p in enumerate(preds)])

    with open('i2w.pkl', 'rb') as f:
        i2w = pickle.load(f)

    for pred in preds:
        out = [i2w[p] for p in pred[0].tolist()]
        print(' '.join(out))
