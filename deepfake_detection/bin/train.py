import tensorflow as tf
import keras
import argparse
import sys
import os

sys.path.append("..")
from callbacks import RedirectModel
from preprocessing import DataGenerator
import models


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_callbacks(model, args):
    """ Creates the callbacks to use during training.
        Args
            model: The base model.
            validation_generator: The generator for creating validation data.
            args: parseargs args object.
        Returns:
            A list of callbacks used for training.
        """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            batch_size=args.batch_size,
        )
        callbacks.append(tensorboard_callback)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{model}_{dataset_type}_{{epoch:02d}}.h5'.format(model=args.model, dataset_type=args.dataset_type)
            ),
            verbose=1
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        patience=2,
        verbose=1
    ))

    return callbacks


def create_generators(args):
    """ Create generators for training and validation.
        Args
            args             : parseargs object containing configuration for generators.
            preprocess_image : Function that preprocesses an image for the network.
        """
    common_args = {
        'batch_size': args.batch_size,
        'config': args.config,
        'image_min_side': args.image_min_side,
        'image_max_side': args.image_max_side,
        # 'preprocess_image': preprocess_image,
    }

    # create random transform generator for augmenting training data
    # if args.random_transform:
    #     transform_generator = random_transform_generator(
    #         min_rotation=-0.1,
    #         max_rotation=0.1,
    #         min_translation=(-0.1, -0.1),
    #         max_translation=(0.1, 0.1),
    #         min_shear=-0.1,
    #         max_shear=0.1,
    #         min_scaling=(0.9, 0.9),
    #         max_scaling=(1.1, 1.1),
    #         flip_x_chance=0.5,
    #         flip_y_chance=0.5,
    #     )
    # else:
    #     transform_generator = random_transform_generator(flip_x_chance=0.5)

    if args.dataset_type == 'csv':
        train_generator = DataGenerator(
            args.annotations,
            shuffle=True,
            is_train=True,
            # args.classes,
            # transform_generator=transform_generator,
            **common_args
        )
        if args.val_annotations:
            validation_generator = DataGenerator(
                args.val_annotations,
                shuffle=True,
                is_train=False,
                **common_args
            )
        else:
            validation_generator = None
    else:
        raise ValueError(
            'Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator
    # return train_generator


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
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights', help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights', help='Initialize the model with weights from a file.')

    parser.add_argument('--model', help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size', help='Size of the batches.', default=6, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=2000)
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-4)
    parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    # parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument(
        '--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=320)
    parser.add_argument(
        '--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=800)
    parser.add_argument(
        '--config', help='Path to a configuration parameters .ini file.')

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

    # optionally load config parameters
    # if args.config:
    #     args.config = read_config_file(args.config)

    # create the generators
    # train_generator, validation_generator = create_generators(args)
    train_generator, validation_generator = create_generators(args)

    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.snapshot)
    else:
        print('Creating model, this may take a second...')
        model = models.create_model(args.model)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=args.lr, clipnorm=0.001), metrics=['accuracy'])

    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(model, args)

    return model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
