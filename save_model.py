import argparse


def main(args):
    import tensorflow as tf
    from deeplab.model import Deeplabv3

    deeplab_model = Deeplabv3(
        classes=2, activation='softmax', model_path=args.model_from,
    )
    deeplab_model.save(args.model_to)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_from")
    parser.add_argument("model_to")
    args = parser.parse_args()

    main(args)
