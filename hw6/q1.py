import os
import pickle
import sys

import numpy as np
from skimage.io import imread, imsave


def process(image):
    img = np.copy(image)
    img = img.reshape([600, 600, 3])
    img -= np.min(img)
    img /= np.max(img)
    img = (img * 255).astype(np.uint8)
    return img


folder = sys.argv[1]
imagename = sys.argv[2]

fname = os.path.join(folder, imagename)

target0 = imread(fname).ravel().astype(np.float64)

with open('avgface.pkl', 'rb') as f:
    avg = pickle.load(f)

with open('eigface.pkl', 'rb') as f:
    eig = pickle.load(f)

target = target0 - avg

weights = np.dot(target, eig.T)
print(weights)

reconstruct = avg + np.dot(weights, eig)

imsave('reconstruct.jpg', process(reconstruct))
# imsave('original.jpg', process(target))

comp = np.hstack([process(target0), process(reconstruct)])
imsave('comparison.jpg', comp)
