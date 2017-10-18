import csv
import sys

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

STEPS = (254 * 10000)
VERSION = 5

feature_columns = [tf.feature_column.numeric_column("x", shape=[106])]

linear = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                       model_dir="model/tf-wide-" +
                                       str(VERSION),
                                       n_classes=2)

dnn = tf.estimator.DNNClassifier(hidden_units=[128, 32],
                                 feature_columns=feature_columns,
                                 model_dir="model/tf-deep-" + str(VERSION),
                                 n_classes=2)

linednn = tf.estimator.DNNLinearCombinedClassifier(model_dir="model/tf-dw-" + str(VERSION),
                                                   linear_feature_columns=feature_columns,
                                                   dnn_feature_columns=feature_columns,
                                                   dnn_hidden_units=[128, 64],
                                                   n_classes=2)

col_filter = []
col_filter.append(list(range(0, 1)))  # age
col_filter.append(list(range(1, 2)))  # fnlwgt
col_filter.append(list(range(2, 3)))  # sex
col_filter.append(list(range(3, 4)))  # capital_gain
col_filter.append(list(range(4, 5)))  # capital_loss
col_filter.append(list(range(5, 6)))  # hours_per_week
col_filter.append(list(range(6, 15)))  # employer
col_filter.append(list(range(15, 22)))  # edu_num
col_filter.append(list(range(22, 31)))  # edu
col_filter.append(list(range(31, 38)))  # maritial
# col_filter.append(list(range(38,53))) # occupation
col_filter.append(list(range(53, 59)))  # relationship
# col_filter.append(list(range(59,64))) # race
col_filter.append(list(range(64, 106)))  # country


def train(X_train, Y_train, est, steps):
    x_d = np.genfromtxt(X_train, dtype=np.float32,
                        skip_header=1, delimiter=',')
    x = {"x": x_d}
    y = np.genfromtxt(Y_train, dtype=np.float32, skip_header=1,
                      delimiter=',', usecols=col_filter)

    train_fn = tf.estimator.inputs.numpy_input_fn(
        x=x, y=y, shuffle=True, num_epochs=None)
    eval_fn = tf.estimator.inputs.numpy_input_fn(
        x=x, y=y, num_epochs=1, shuffle=False)

    if est == "linear":
        linear.train(train_fn, steps=1000000)
        print(linear.evaluate(eval_fn))
    elif est == "dnn":
        dnn.train(train_fn, steps=1000000)
        print(dnn.evaluate(eval_fn))
    elif est == "linednn":
        linednn.train(train_fn, steps=1000000)
        print(linednn.evaluate(eval_fn))


def test(X_test, est, outputFile):
    t_d = np.genfromtxt(X_test, dtype=np.float32,
                        skip_header=1, delimiter=',', usecols=col_filter)
    t = {"x": t_d}
    test_fn = tf.estimator.inputs.numpy_input_fn(x=t, shuffle=False)
    if est == "linear":
        pred = linear.predict(test_fn)
        arr = [p for p in pred]
    elif est == "dnn":
        pred = dnn.predict(test_fn)
        arr = [p for p in pred]
    elif est == "linednn":
        pred = linednn.predict(test_fn)
        arr = [p for p in pred]

    with open(outputFile, 'w') as of:
        w = csv.writer(of)
        w.writerow(["id", "label"])
        for i, x in enumerate([p["class_ids"].tolist() for p in arr]):
            w.writerow([i + 1, x[0]])


if __name__ == "__main__":
    mode = sys.argv[1]
    est = sys.argv[2]

    if mode == "train":
        X = sys.argv[3]
        Y = sys.argv[4]
        train(X, Y, est, STEPS)
    else:
        T = sys.argv[3]
        outputFile = sys.argv[4]
        test(T, est,  outputFile)
