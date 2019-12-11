import os
import numpy as np
import cv2
import argparse
from glob import glob
from tqdm import tqdm

def split_folder(video_folder_path, ds):
    videos = glob(video_folder_path+'/*.mp4')
    for video in tqdm(videos):
        filename = video.split('/')[-1].split('.')[0]
        saved_loc = '{}/{}'.format(ds,filename)
        try :
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

    if args.isfolder:
        if args.folder == None:
            return
        split_folder(args.folder, args.dir)
    else:
        split_into_frames(args.vid, args.dir)


if __name__ == '__main__':
    # example : python spliter.py a images True hello 로 실행하면 hello 폴더의 동영상을 다 읽어서 images 안에 해당 동영상이름으로 폴더를 만들고 거기에 파싱한 이미지를 다 넣음.
    # example : python spliter.py dd.mp4 images 로 실행하면 하나의 동영상 파일 dd.mp4가 파싱되어 images 폴더안에 저장됨


    parser = argparse.ArgumentParser()
    parser.add_argument('vid',help='video filename',)
    parser.add_argument('dir',help='folder path for saving images',)
    parser.add_argument('isfolder', type=bool, default=False)
    parser.add_argument('folder', default = None)

    args = parser.parse_args()
    main(args)
