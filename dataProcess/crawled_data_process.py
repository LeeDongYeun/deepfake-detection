# python crawled_data_process.py --imgdir /root/data/crawled_images_resized --csvdir /root/csv/train
import pandas as pd
import glob
import os
import random
import argparse


def main(config):
    parents = glob.glob(config.imgdir + "/*")

    train = []

    random.shuffle(parents)
    for p in parents:
        train.append([p, 1])

    train = pd.DataFrame(train)
    train = train.reset_index(drop=True)

    train.to_csv('{}/crawled_train.csv'.format(
        config.csvdir), index=False, header=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--imgdir', help='image root directory')
    parser.add_argument('--csvdir', help='csv train directory')

    config = parser.parse_args()

    main(config)
