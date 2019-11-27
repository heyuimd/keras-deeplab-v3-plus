import os
import numpy as np
from PIL import Image
from deeplab.model import Deeplabv3
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import cv2
import random
import argparse
import math


class DataGenerator(Sequence):

    def __init__(self, x, y, data_type='train', batch_size=8,
                 blur=0, horizontal_flip=True,
                 rotation=45.0, zoom=0.2):

        # sanity check
        assert x.shape[0] == y.shape[0]
        assert data_type in ['train', 'val']

        self.x = x
        self.y = y
        self.len = x.shape[0]
        self.data_type = data_type
        self.batch_size = batch_size
        self.blur = blur
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
        self.zoom = zoom

    def __len__(self):
        return math.ceil(self.len / self.batch_size)

    def __getitem__(self, idx):
        x_list = []
        y_list = []

        for (x, y) in zip(
            self.x[idx*self.batch_size:
                   (idx+1)*self.batch_size].astype('float32'),
            self.y[idx*self.batch_size:
                   (idx+1)*self.batch_size].astype('float32'),
        ):
            if self.data_type == 'train':
                if self.blur and random.randint(0, 1):
                    x = cv2.GaussianBlur(x, (self.blur, self.blur), 0)

                # Do augmentation
                if self.horizontal_flip and random.randint(0, 1):
                    x = cv2.flip(x, 1)
                    y = cv2.flip(y, 1)

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
                        (x.shape[1]//2, x.shape[0]//2), angle, scale)
                    x = cv2.warpAffine(
                        x, M, (x.shape[1], x.shape[0]), borderMode=cv2.BORDER_REFLECT)
                    y = cv2.warpAffine(
                        y, M, (y.shape[1], y.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

            x = (x / 127.5) - 1.
            y = y[:, :, np.newaxis]

            x_list.append(x.copy())
            y_list.append(y.copy())

            X = np.empty(
                (len(x_list), x_list[0].shape[0], x_list[0].shape[1], 3))
            Y = np.empty(
                (len(y_list), y_list[0].shape[0], y_list[0].shape[1], 1))

            for i, (x, y) in enumerate(zip(x_list, y_list)):
                X[i] = x
                Y[i] = y

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

    t = DataGenerator(*get_data(args.data_dir, 1, 100), data_type='train')
    v = DataGenerator(*get_data(args.data_dir, 100, None), data_type='val')

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
        epochs=200,
        validation_data=v,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='the directory for data')
    args = parser.parse_args()

    main(args)
