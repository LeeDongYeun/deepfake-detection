import cv2
import argparse
import glob
import os
import pandas as pd
import random
import csv
import multiprocessing as mp


def to_csv(paths, csv_path):
    csvFile = open(csv_path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(csvFile)
    for path in paths:
        wr.writerow([path, 1]) 
    csvFile.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--original_path',
			    type=str,
			    nargs='+')
	parser.add_argument('--csv_path',
			    help='csv file name',
			    type=str,
			    default='true_dataset.csv')
	args = parser.parse_args()
	paths = [file for path in args.original_path for file in glob.glob(path + '/*')]
    
	to_csv(paths, args.csv_path)
