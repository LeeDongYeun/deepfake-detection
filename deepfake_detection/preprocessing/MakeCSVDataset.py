import pandas as pd
import glob
import os
import random
import argparse


def main(config):
    parents = glob.glob(config.dir + "/*")

    train = []
    val = []

    random.shuffle(parents)
    thresh = int(len(parents)*config.ratio)
    for i, p in enumerate(parents):
        if i < thresh:
            train.append([p, 1])
        else:
            val.append([p, 1])

    train = pd.DataFrame(train)
    val = pd.DataFrame(val)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train.to_csv('{}_train.csv'.format(
        config.filename), index=False, header=False)
    val.to_csv('{}_val.csv'.format(config.filename), index=False, header=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dir', help='image root directory')
    parser.add_argument(
        '--ratio', help='train, validation dataset split ratio', type=float, default=0.8)
    parser.add_argument(
        '--filename', help='csv header name', default='dataset')

    config = parser.parse_args()

    main(config)
