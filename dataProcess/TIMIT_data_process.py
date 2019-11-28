# 동영상 쪼개는 command
# python TIMIT_data_process.py --imgdir /root/data/deepfakeTIMIT --viddir /root/rawData/DeepfakeTIMIT --split True
# 쪼개진 동영상 이미지들 경로 csv로 만드는 command
# python TIMIT_data_process.py --imgdir /root/data/deepfakeTIMIT --csvdir /root/csv

import os
import numpy as np
import cv2
import argparse
from glob import glob
from tqdm import tqdm
import pandas as pd
import random


def split_folder(video_folder_path, ds):
    videos = glob(video_folder_path+'/*.avi')
    for video in tqdm(videos):
        filename = video.split('/')[-1].split('.')[0]
        saved_loc = '{}/{}'.format(ds, filename)
        try:
            os.mkdir(saved_loc)
        except:
            print("can't make directory")
        print(videos, saved_loc)
        split_into_frames(video, saved_loc)
        print("Video converted to images : {}".format(video))
    print('------------finish--------------')


def split_into_frames(video_path, folder):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    while True:
        success, image = vidcap.read()
        if success:
            imgs.append(image)
        else:
            break

    vidcap.release()
    if len(imgs) != frame_num:
        frame_num = len(imgs)
    for id, im in enumerate(imgs):
        im_name = '{:05d}.jpg'.format(id)
        cv2.imwrite(folder + '/' + im_name, im)
    print('Save original images to folder {}'.format(folder))


def to_csv(imgdir, csvdir, ratio):
    parents = glob(imgdir + "/*/*/*/*")

    train = []
    val = []

    random.shuffle(parents)
    thresh = int(len(parents)*ratio)
    for i, p in enumerate(parents):
        if i < thresh:
            train.append([p, 1])
        else:
            val.append([p, 1])

    train = pd.DataFrame(train)
    val = pd.DataFrame(val)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train.to_csv('{}/train/timit_train.csv'.format(csvdir),
                 index=False, header=False)
    val.to_csv('{}/val/timit_val.csv'.format(csvdir),
               index=False, header=False)


def main(args):
    if args.split:
        # for higher_quality folder
        imgdir = args.imgdir+'/higher_quality'
        viddir = args.viddir+'/higher_quality'
        folders = glob(viddir+'/*')
        for fd in folders:
            fdname = fd.split('/')[-1].split('.')[0]
            saved_loc = '{}/{}'.format(imgdir, fdname)
            try:
                os.mkdir(saved_loc)
            except:
                print("fd can't make directory")
            split_folder(fd, saved_loc)
        # for lower_quality folder
        imgdir = args.imgdir+'/lower_quality'
        viddir = args.viddir+'/lower_quality'
        folders = glob(viddir+'/*')
        for fd in folders:
            fdname = fd.split('/')[-1].split('.')[0]
            saved_loc = '{}/{}'.format(imgdir, fdname)
            try:
                os.mkdir(saved_loc)
            except:
                print("fd can't make directory")
            split_folder(fd, saved_loc)
    else:
        to_csv(args.imgdir, args.csvdir, args.ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', help='timit images directory')
    parser.add_argument('--viddir', help='video directory', default='')
    parser.add_argument('--csvdir', help='csv directory', default='')
    parser.add_argument('--split', default=False)
    parser.add_argument(
        '--ratio', help='train, validation dataset split ratio', type=float, default=0.8)

    args = parser.parse_args()
    main(args)
