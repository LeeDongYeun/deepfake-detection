import tensorflow as tf
import keras
import argparse
import sys
import os

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import deepfake_detection.bin  # noqa: F401
    __package__ = "deepfake_detection.bin"

from ..callbacks import RedirectModel
from ..preprocessing import DataGenerator
from .. import models

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def create_generators(args):
    """ Create generators for training and validation.
        Args
            args             : parseargs object containing configuration for generators.
            preprocess_image : Function that preprocesses an image for the network.
        """
    common_args = {
        'batch_size': args.batch_size,
        'config': args.config,
        # 'preprocess_image': preprocess_image,
    }

    if args.dataset_type == 'csv':
        validation_generator = DataGenerator(
            args.annotations,
            shuffle=True,
            is_train=False,
            **common_args
        )
    else:
        raise ValueError(
            'Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(
        help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('weights', help='Path to evaluate model weight.')
    parser.add_argument('--model', help='Model name.', default='resnet50', type=str)
    parser.add_argument('--batch-size', help='Size of the batches.', default=16, type=int)    
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    # parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())


    validation_generator = create_generators(args)
    
    print('Loading model, this may take a second...')
    model = models.load_model(args.weights)

    print(model.summary())

    scores = model.evaluate_generator(validation_generator)
    print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

    return scores


if __name__ == '__main__':
    main()
