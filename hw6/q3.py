import sys
import csv
import numpy as np
from sklearn.cluster import KMeans

from keras.layers import Input, Reshape, Conv2D, Flatten, Dense, Conv2DTranspose
from keras.models import Model

input_img = Input(shape=(784, ))

encoded = Reshape([28, 28, 1])(input_img)
encoded = Conv2D(128, 3, strides=2, padding='same', activation='relu')(encoded)
encoded = Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoded)
encoded = Flatten()(encoded)
encoded = Dense(196, activation='tanh')(encoded)

decoded = Reshape([7, 7, -1])(encoded)
decoded = Conv2DTranspose(
    64, 4, strides=2, padding='same', activation='relu')(decoded)
decoded = Conv2DTranspose(
    1, 4, strides=2, padding='same', activation='sigmoid')(decoded)
decoded = Flatten()(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# autoencoder.summary()
autoencoder.load_weights('ae6')

images = np.load(sys.argv[1])

cases = []
with open(sys.argv[2]) as f:
    re = csv.reader(f)
    next(re)
    for row in re:
        cases.append([int(row[1]), int(row[2])])

encoder = Model(input_img, encoded)
reduced = encoder.predict(images)
labels = KMeans(2).fit_predict(reduced)

with open(sys.argv[3], 'w') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'Ans'])
    for i, (r1, r2) in enumerate(cases):
        if labels[r1] == labels[r2]:
            w.writerow([i, 1])
        else:
            w.writerow([i, 0])
