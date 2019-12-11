import pandas as pd
import glob
import os
import random
import argparse
import csv

def to_csv(path, filename, ratio):
    paths = os.listdir(path)
    all_path = []

    for p in paths:
        if p == 'real':
            label = 1
        else:
            label = 0
        
        abs_p = path + '/' + p + '/'
        videos = os.listdir(abs_p)
        for video in videos:
            abs_video = abs_p + video + '/'
            imgs = os.listdir(abs_video)

            for img in imgs:
                abs_img = abs_video + img
                all_path.append([abs_img, label])
    
    random.shuffle(all_path)
    thresh = int(len(all_path)*ratio)
    
    train = []
    val = []

    for i, p in enumerate(all_path):
        if i < thresh:
            train.append(p)
        else:
            val.append(p)
    
    train = pd.DataFrame(train)
    val = pd.DataFrame(val)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train.to_csv('{}_train.csv'.format(filename), index=False, header=False)
    val.to_csv('{}_val.csv'.format(filename), index=False, header=False)


def main(args):
    to_csv(args.dir, args.filename, args.ratio)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Process UADFV dataset')

    parser.add_argument('--dir', help='UADFV dataset directory')
    parser.add_argument(
        '--ratio', help='train, validation dataset split ratio', type=float, default=0.8)
    parser.add_argument(
        '--filename', help='csv file name', default='UADFV')

    args = parser.parse_args()

    main(args)
