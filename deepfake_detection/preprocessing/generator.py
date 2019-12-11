import os
import argparse

# getting data from the csv file which is form of "image link" "real or not"
import csv
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
import random

# keras data generator
import numpy as np
import keras

# Blurring process
import cv2


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, csv_path, shuffle, is_train, **kwargs):
        super(DataGenerator, self).__init__()
        self.shuffle = shuffle
        self.batch_size = kwargs['batch_size']
        self.csv_path = csv_path
        self.data, self.label = self._read_csv(self.csv_path)
        self.is_train = is_train

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.label) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X = self.data[index*self.batch_size:(index+1)*self.batch_size]
        X = self._convert_img(X)
        Y = self.label[index*self.batch_size:(index+1)*self.batch_size]
        if self.is_train:
            varied_X = []
            varied_Y = []
            for i, x in enumerate(X):
                rand = random.random()
                if rand > 0.5:
                    varied_X.append(x)
                    varied_Y.append(Y[i])
                else:
                    if Y[i] == 0:
                        varied_X.append(x)
                        varied_Y.append(Y[i])
                    else:
                        varied_X.append(self._transform_function(x))
                        varied_Y.append(1)
        else:
            varied_X = X
            varied_Y = Y

        return np.array(varied_X), np.array(varied_Y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        np.random.shuffle(self.indexes)

    def _convert_img(self, path_list):
        data = []
        for path in path_list:
            img = load_img(path)
            d = img_to_array(img)
            d = cv2.resize(d, dsize=(256,256))

            data.append(d)

        return data

    def _read_csv(self, path):
        data, label = [], []
        with open(path, newline='') as f:
            reading = list(csv.reader(f, delimiter=' ', quotechar='|'))
            for row in reading:
                p, r = row[0].split(",")
                data.append(p)
                label.append(r)

        return data, label

    def _transform_function(self, img):
        face_cascade = cv2.CascadeClassifier(
            os.getcwd()+'/haarcascade_frontface.xml')  # data provided from open cv
        gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5)  # detecting face from received img
        
        fail = 0
        
        # print(type(faces))
        if hasattr(faces, 'shape'):
            for (x, y, w, h) in faces:
                cropimg = img[y:y+h, x:x+w]

            alsize = random.randrange(1, 10)
            fx = 0.6 + 0.1*alsize
            fy = 0.6 + 0.1*alsize

            alimg = cv2.resize(cropimg, dsize=(0, 0), fx=fx,
                               fy=fy, interpolation=cv2.INTER_LINEAR)
            temp = cv2.GaussianBlur(alimg, (5, 5), 0)  # 얼굴부분 추출해서 Gaussian Blur
            blur = cv2.resize(temp, dsize=(0, 0), fx=1/fx, fy=1/fy,
                              interpolation=cv2.INTER_LINEAR)

            for i in range(y, y+h):
                for j in range(x, x+w):
                    for k in range(3):
                        try:
                            img[i, j, k] = blur[i-y, j-x, k]
                        except:
                            # print("fail")
                            fail += 1
                            continue
        
        if fail != 0:
            print(fail)
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',
                        help='path of the csv file',
                        type=str)
    parser.add_argument('--batch_size',
                        help='number of batch',
                        type=int,
                        default=4)
    args = parser.parse_args()

    def depath(path): return os.path.realpath(os.path.expanduser(path))
    csv_path = depath(args.csv_path)
    d = {'batch_size': args.batch_size}

    training_generator = DataGenerator(args.csv_path, True, **d)

    a = training_generator
    x, y = a.__getitem__(1)
    save_img('1.jpg', array_to_img(x[0]))
    save_img('2.jpg', array_to_img(x[1]))
    save_img('3.jpg', array_to_img(x[2]))
    save_img('4.jpg', array_to_img(x[3]))
