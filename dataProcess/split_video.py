import os
import numpy as np
import cv2
import argparse

def split_into_frames(video_path, folder):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
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

    
def main(args):
    print(args.dir)
    split_into_frames(args.vid, args.dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid',help='video filename',)
    parser.add_argument('--dir',help='folder path for saving images',)

    args = parser.parse_args()
    main(args)