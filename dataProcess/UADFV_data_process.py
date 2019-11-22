import pandas as pd
import glob
import os
import random
import argparse
import csv

def to_csv(path, csv_path):
    csvFile = open(csv_path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(csvFile)
    paths = os.listdir(path)

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
                wr.writerow([abs_img, label])
    
    csvFile.close()

def main(args):
    to_csv(args.dir, args.filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Process UADFV dataset')

    parser.add_argument('--dir', help='UADFV dataset directory')
    parser.add_argument(
        '--ratio', help='train, validation dataset split ratio', type=float, default=0.8)
    parser.add_argument(
        '--filename', help='csv file name', default='UADFV_dataset.csv')

    args = parser.parse_args()

    main(args)