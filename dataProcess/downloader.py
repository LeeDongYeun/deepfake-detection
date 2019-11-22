import pandas as pd
import requests
import argparse
from tqdm import tqdm


def download(url, file_name):
    with open(file_name, "wb") as file:   # open in binary mode
        response = requests.get(url)      # get request
        file.write(response.content)

def main(args):
    
    df = pd.read_csv(args.csv)
    
    for index, row in tqdm(df.iterrows()):
        if index % 1000 == 0:
            print(index, "images are downloaded")
        if index>args.num:
            break
        url = row[0]
        filename = '{}/{}.jpg'.format(args.dir,url.split('/')[-2])
        #print(filename)
        download(url, filename)
    
    print(index, "images are downloaded")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv', help='csv filename', default = 'face_1-95.csv')
    parser.add_argument('--num', help='number of images to download',type = int, default = 10000)
    parser.add_argument('--dir', help='images download path', default = 'images')

    args = parser.parse_args()
    main(args)
