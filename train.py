import os
import numpy as np
from PIL import Image
from deeplab.model import Deeplabv3
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import cv2
import random
import argparse


class TrainGenerator(Sequence):

    def __init__(self, x, y, seed=7,
                 blur=0, horizontal_flip=True, vertical_flip=True,
                 rotation=45.0, zoom=0.2):

        # sanity check
        assert x.shape[0] == y.shape[0]

        np.random.seed(seed)

        self.x = x
        self.y = y
        self.len = x.shape[0]
        self.idx_now = 0
        self.blur = blur
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.zoom = zoom

    def __len__(self):
        return self.len

    def __getitem__(self, i):

        X = self.x[self.idx_now].astype('float32')
        Y = self.y[self.idx_now].astype('float32')

        if self.blur and random.randint(0, 1):
            X = cv2.GaussianBlur(X, (self.blur, self.blur), 0)

        # Do augmentation
        if self.horizontal_flip and random.randint(0, 1):
            X = cv2.flip(X, 1)
            Y = cv2.flip(Y, 1)

        if self.vertical_flip and random.randint(0, 1):
            X = cv2.flip(X, 0)
            Y = cv2.flip(Y, 0)

        if self.rotation:
            angle = random.gauss(mu=0.0, sigma=self.rotation)
        else:
            angle = 0.0

        if self.zoom:
            scale = random.gauss(mu=1.0, sigma=self.zoom)
        else:
            scale = 1.0

        if self.rotation or self.zoom:
            M = cv2.getRotationMatrix2D(
                (X.shape[1]//2, X.shape[0]//2), angle, scale)
            X = cv2.warpAffine(
                X, M, (X.shape[1], X.shape[0]), borderMode=cv2.BORDER_REFLECT)
            Y = cv2.warpAffine(
                Y, M, (Y.shape[1], Y.shape[0]), borderMode=cv2.BORDER_REFLECT)

        self.idx_now = (self.idx_now + 1) % self.len

        X = (X / 127.5) - 1.
        X = X[np.newaxis]
        Y = Y[np.newaxis, :, :, np.newaxis]

        return X, Y


class ValidateGenerator(Sequence):

    def __init__(self, x, y):

        # sanity check
        assert x.shape[0] == y.shape[0]

        self.x = x
        self.y = y
        self.len = x.shape[0]
        self.idx_now = 0

    def __len__(self):
        return self.len

    def __getitem__(self, i):

        X = self.x[self.idx_now].astype('float32')
        Y = self.y[self.idx_now].astype('float32')

        self.idx_now = (self.idx_now + 1) % self.len

        X = (X / 127.5) - 1.
        X = X[np.newaxis]
        Y = Y[np.newaxis, :, :, np.newaxis]

        return X, Y


def get_data(data_dir, start=None, end=None):
    images = None
    masks = None
    first = True

    if start is None:
        start = 1
    if end is None:
        end = 118

    for i in range(start, end + 1):
        filename = f'{i:03d}.npy'
        if first:
            first = False
            im = np.load(os.path.join(data_dir, 'images', filename))
            images = im[np.newaxis]
            im = np.load(os.path.join(data_dir, 'seg', filename))
            masks = im[np.newaxis]
        else:
            im = np.load(os.path.join(data_dir, 'images', filename))
            images = np.concatenate([images, im[np.newaxis]])
            im = np.load(os.path.join(data_dir, 'seg', filename))
            masks = np.concatenate([masks, im[np.newaxis]])

    return images, masks


def main(args):
    model = Deeplabv3(classes=2, activation='softmax')
    model.summary()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    t = TrainGenerator(*get_data(args.data_dir, 1, 100))
    v = ValidateGenerator(*get_data(args.data_dir, 100, None))

    callbacks = [
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model-{epoch:02d}.h5',
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=4,
            factor=0.1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=9
        ),
    ]

    model.fit_generator(
        t,
        epochs=100,
        validation_data=v,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='the directory for data')
    args = parser.parse_args()

    main(args)
