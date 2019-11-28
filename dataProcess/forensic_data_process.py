# python forensic_data_process.py --imgdir /root/data/faceforensics_deepfakes --csvdir /root/csv

import pandas as pd
import glob
import os
import random
import argparse


def main(config):
    parents = glob.glob(config.imgdir + "/*/*")

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

    train.to_csv('{}/train/forensic_train.csv'.format(
        config.csvdir), index=False, header=False)
    val.to_csv('{}/val/forensic_val.csv'.format(config.csvdir),
               index=False, header=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--imgdir', help='image root directory')
    parser.add_argument('--csvdir', help='csv directory')

    parser.add_argument(
        '--ratio', help='train, validation dataset split ratio', type=float, default=0.8)

    config = parser.parse_args()

    main(config)
