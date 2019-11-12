import os
import argparse
import multiprocessing
from multiprocessing import Pool

# getting data from the csv file which is form of "image link" "real or not"
import csv
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import random

# keras data generator
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, csv_path, **kwargs):
        super(DataGenerator, self).__init__()
        # self.data = data
        # self.label = label
        self.batch_size = kwargs['batch_size']
        self.csv_path = csv_path
        _, __, self.data, self.label = self.read_csv(self.csv_path)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.label) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X = self.data[index*self.batch_size:(index+1)*self.batch_size]
        varied_X = list(map(lambda x: x if random.random() > 0.5 else None, X))
        print(varied_X)
        varied_indexes = list(
            map(lambda x: 1 if hasattr(x, 'shape') else 0, varied_X))
        varied_X = list(filter(lambda x: hasattr(x, 'shape'), varied_X))
        # varied_X = self._apply_fn(varied_X)
        # varied_X = list(map(lambda x, y, z: y if z else x, X, varied_X, varied_indexes))
        Y = self.label[index*self.batch_size:(index+1)*self.batch_size]
        # varied_Y = list(map(lambda x, y: y if x == 0 else 0, varied_indexes, Y))
        # return varied_X, varied_Y
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
        np.random.shuffle(self.indexes)

    def _apply_fn(self, array_list):
        array_list = list(map(lambda x: array_to_img(x), array_list))
        array_list = list(map(lambda x: transform_function(x), array_list))
        array_list = list(map(lambda x: img_to_array(x), array_list))

        return array_list

    def convert_img_fn(self, path):
        img = load_img(path)
        img_array = img_to_array(img)

        return img_array


    def convert_img_mp(self, cpu_ccount, path_list):
        pool = Pool(processes=cpu_ccount)
        path_list = list(map(lambda x: '../.'+x, path_list))
        r = pool.map_async(self.convert_img_fn, path_list)
        r.wait()
        pool.close()
        pool.join()
        data = r.get()

        return data


    def convert_img(self, path_list):
        ncpus = multiprocessing.cpu_count()
        if ncpus > 1:
            data = self.convert_img_mp(ncpus, path_list)
        else:
            data = []
            for path in path_list:
                d = self.convert_img_fn(path)
                data.append(d)

        return data


    def read_csv(self, path):
        data, real = [], []
        with open(path, newline='') as f:
            r = list(csv.reader(f, delimiter=' ', quotechar='|'))[:21]
            for row in r[1:]:
                p, r = row[0].split(",")
                data.append(p)
                real.append(r)

        data = self.convert_img(data)

        l = list(range(len(data)))
        ids = list(map(lambda x: 'id-'+str(x), l))
        random.shuffle(ids)
        nt = int(len(data) * 0.8)
        nv = int(len(data) * 0.2)
        train = ids[:nt]
        val = ids[nt:nt+nv]
        test = ids[nt+nv:]
        trainp, valp, testp, traind, vald, testd = [], [], [], [], [], []
        for idx, d in enumerate(data):
            if 'id-'+str(idx) in train:
                trainp.append(d)
                traind.append(real[idx])
            elif 'id-'+str(idx) in val:
                valp.append(d)
                vald.append(real[idx])
            else:
                testp.append(d)
                testd.append(real[idx])

        pd = {'train': train, 'validation': val, 'test': test}
        dd = {'train': trainp, 'validation': valp, 'test': testp}
        ld = {'train': traind, 'validation': vald, 'test': testd}

        return d, pd, dd, ld
