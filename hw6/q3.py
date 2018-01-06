import sys
import csv
import numpy as np
from sklearn.cluster import KMeans
import pickle

from keras.layers import Input, Reshape, Conv2D, Flatten, Dense, Conv2DTranspose
from keras.models import Model, load_model

input_img = Input(shape=(784, ))

encoded = Dense(392, activation='relu')(input_img)
encoded = Dense(196, activation='relu')(encoded)
encoded = Dense(49)(encoded)

# decoded = Dense(196, activation='relu')(encoded)
# decoded = Dense(392, activation='relu')(decoded)
# decoded = Dense(784, activation='sigmoid')(decoded)

# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# autoencoder.summary()

# # autoencoder.summary()
# autoencoder.load_weights('model/ae-2')

images = np.load(sys.argv[1])

cases = []
with open(sys.argv[2]) as f:
    re = csv.reader(f)
    next(re)
    for row in re:
        cases.append([int(row[1]), int(row[2])])

encoder = Model(input_img, encoded)
encoder.compile('adam', 'mse')
encoder.load_weights('enc')
reduced = encoder.predict(images)
# with open('model/km.pkl', 'rb') as f:
#     km = pickle.load(f)
km = KMeans(2)
labels = km.fit_predict(reduced)

with open(sys.argv[3], 'w') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'Ans'])
    for i, (r1, r2) in enumerate(cases):
        if labels[r1] == labels[r2]:
            w.writerow([i, 1])
        else:
            w.writerow([i, 0])
