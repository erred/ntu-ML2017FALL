import pickle
import sys

import numpy as np
import xgboost

import iofn

MODELFILE = 'model/xg-2'

def main():
    mode = sys.argv[1]
    modelFile = MODELFILE
    X = sys.argv[2]
    T = sys.argv[3]
    Y = sys.argv[4]

    x, t, y = iofn.readData(X, T, Y)

    if mode == "train":
        order = np.arange(len(x))
        np.random.shuffle(order)
        split = int(len(x)*0.9)
        x_t = x[order][:split]
        y_t = y[order][:split]
        x_e = x[order][split:]
        y_e = y[order][split:]

        xg = xgboost.XGBClassifier(
            max_depth=10,
            learning_rate=0.1,
            n_estimators=100,
            n_jobs=4)

        xg.fit(x_t, y_t)
        with open(MODELFILE, 'wb') as fo:
            pickle.dump(xg, fo)
        preds = xg.predict(x_e)
        print(float(sum([1 if p == y_e[i] else 0 for i, p in enumerate(preds)]))/len(preds))

    if mode == "eval":
        with open(MODELFILE, 'rb') as fo:
            xg = pickle.load(fo)
        preds = xg.predict(x)
        print(float(sum([1 if p == y[i] else 0 for i, p in enumerate(preds)]))/len(preds))
    if mode == "test":
        outputFile = sys.argv[5]
        with open(MODELFILE, 'rb') as fo:
            xg = pickle.load(fo)
        preds = xg.predict(t)
        iofn.saveOutput([[i+1, int(x)] for i, x in enumerate(preds)],outputFile)




if __name__ == "__main__":
    main()
