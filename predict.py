import numpy as np
from PIL import Image
import cv2
import argparse


def main(args):
    from deeplab.model import Deeplabv3

    trained_image_width = args.dim
    mean_subtraction_value = 127.5
    image = np.array(Image.open(args.input_path))

    # resize to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype(
        'uint8')).resize((int(ratio * h), int(ratio * w))))

    # apply normalization for trained dataset images
    resized_image = (resized_image / mean_subtraction_value) - 1.

    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(
        resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    # make prediction
    deeplab_model = Deeplabv3(
        classes=2, activation='softmax', model_path=args.model_path)
    res = deeplab_model.predict(np.expand_dims(resized_image, 0))
    labels = np.argmax(res.squeeze(), -1)

    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

    img_result = np.ones_like(labels)
    img_result[labels == 1] = 0
    cv2.imwrite(args.output_path, img_result*255)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument('-d', '--dim', type=int, default=512,
                        help='dimensions for images used in training')
    args = parser.parse_args()

    main(args)
