import csv
import sys

import numpy as np
import tensorflow as tf

import iofn

tf.logging.set_verbosity(tf.logging.INFO)

col_filter = []
# col_filter.append(list(range(0, 1)))  # age
# col_filter.append(list(range(1, 2)))  # fnlwgt
# col_filter.append(list(range(2, 3)))  # sex
# col_filter.append(list(range(3, 5)))  # capital gain/loss
# col_filter.append(list(range(5, 6)))  # hours_per_week
# col_filter.append(list(range(6, 15)))  # employer
# col_filter.append(list(range(15, 22)))  # edu_num
# col_filter.append(list(range(22, 31)))  # edu
# col_filter.append(list(range(31, 38)))  # maritial
# col_filter.append(list(range(38,53))) # occupation
col_filter.append(list(range(53, 59)))  # relationship
# col_filter.append(list(range(59,64))) # race
col_filter.append(list(range(64, 106)))  # country
col_filter = sum(col_filter, [])

FEATURES = len(col_filter)
EPOCHS = 100
VERSION = 6

feature_columns = [tf.feature_column.numeric_column("x", shape=[FEATURES])]


def createEst(est):
    if est == "linear":
        return tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
            model_dir="model/tf-wide-" +
            str(VERSION),
            n_classes=2)
    elif est == "dnn":
        return tf.estimator.DNNClassifier(
            hidden_units=[128, 32],
            feature_columns=feature_columns,
            model_dir="model/tf-deep-" + str(VERSION),
            n_classes=2)
    elif est == "linednn":
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir="model/tf-dw-" + str(VERSION),
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=[128, 64],
            n_classes=2)

def train(X_train, Y_train, est):
    x_d, t_d, y_d = iofn.readData(X_train, X_test, Y_train, col_filter)
    x = {"x": x_d}

    train_fn = tf.estimator.inputs.numpy_input_fn(
        x=x, y=y_d, shuffle=True, num_epochs=EPOCHS)
    eval_fn = tf.estimator.inputs.numpy_input_fn(
        x=x, y=y_d, num_epochs=1, shuffle=False)

    est.train(train_fn)
    print(est.evaluate(eval_fn))


def test(X_test, est, outputFile):
    x_d, t_d, y_d = iofn.readData(X_train, X_test, Y_train, col_filter)
    t = {"x": t_d}

    test_fn = tf.estimator.inputs.numpy_input_fn(x=t, shuffle=False)
    pred = est.predict(test_fn)
    arr = [p for p in pred]

    with open(outputFile, 'w') as of:
        w = csv.writer(of)
        w.writerow(["id", "label"])
        for i, x in enumerate([p["class_ids"].tolist() for p in arr]):
            w.writerow([i + 1, x[0]])


if __name__ == "__main__":
    mode = sys.argv[1]
    est = createEst(sys.argv[2])

    if mode == "train":
        X = sys.argv[3]
        Y = sys.argv[4]
        train(X, Y, est)
    else:
        T = sys.argv[3]
        outputFile = sys.argv[4]
        test(T, est,  outputFile)
