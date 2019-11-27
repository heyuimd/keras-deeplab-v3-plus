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

    def __init__(self, data_dir, batch_size=32, shape=(512, 512)):
        img_dir = os.path.join(data_dir, 'img')
        seg_dir = os.path.join(data_dir, 'seg')
        list_path = os.path.join(data_dir, 'list.txt')

        # sanity check
        assert os.path.isdir(img_dir)
        assert os.path.isdir(seg_dir)
        assert os.path.isfile(list_path)

        self.batch_size = batch_size
        with open(list_path) as fin:
            file_list = [line.strip() for line in fin]
        self.len = len(file_list)

        self.x = np.empty((self.len, shape[0], shape[1], 3), dtype=np.uint8)
        self.y = np.empty((self.len, shape[0], shape[1], 1), dtype=np.uint8)

        for i, filename in enumerate(file_list):
            x = np.load(os.path.join(img_dir, filename + '.npy'))
            y = np.load(os.path.join(seg_dir, filename + '.npy'))

            self.x[i] = x
            self.y[i] = y[:, :, np.newaxis]

    def __len__(self):
        return math.ceil(self.len / self.batch_size)

    def __getitem__(self, idx):
        X = self.x[idx*self.batch_size:(idx+1)*self.batch_size].astype(np.float32)
        Y = self.y[idx*self.batch_size:(idx+1)*self.batch_size].astype(np.float32)

        X = (X / 127.5) - 1.

        return X, Y


def main(args):
    model = Deeplabv3(classes=21, activation='softmax',
                      alpha=0.25, weights=None)
    model.summary()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    t = DataGenerator(args.train_data_dir)
    v = DataGenerator(args.val_data_dir)

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
        epochs=1000,
        validation_data=v,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_dir')
    parser.add_argument('val_data_dir')
    args = parser.parse_args()

    main(args)
