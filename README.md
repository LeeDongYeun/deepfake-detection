# Deepfake-detection

> KAIST 2019 Fall, CS489 final project

# Table of Contents

- Introduction
- Installation
    - Clone
    - Setup
    - Dataset
    - Pre-trained models
- Code review
- Usage
    - Train
    - Test

# Introduction

This is keras implementation of deepfake-detection based on detecting face warping artifacts, 2019.

# Installation

- All the **code** required to get started.

## Clone

- Clone this repo to your local machine using below command

    $ git clone https://github.com/LeeDongYeun/deepfake-detection

## Setup

> update and install package below

- Tensorflow 1.14  conda install tensorflow-gpu=1.14
- Keras 2.2.4 conda intall keras-gpu=2.2.4
- CUDA=10.1
- OpenCV2
- Numpy
- Matplotlib
- pandas
- tqdm

## Dataset

> Utilized datasets are shown at below link.

### Unsplash api

* A cautionary note is an image based on the search results and may contain unwanted pictures. (For example, a photo covering one side of the face) In such a case, you should delete unwanted pictures directly after downloading them.

This part is divided into two levels.

1. **crawl image path into csv**

    Requirement : node.js

    We use unsplash pictures that appeared when we searched "face". 

    First, to use the api of unsplash, you must register as an unsplash developer and get a token and write the token in getUrls.js.  [https://unsplash.com/developers](https://unsplash.com/developers)

    Second, install node-fetch and fs using npm.

    Third, run getUrls.js with keyword and start page (default 0). we use keyword "face".

    When you run this below code, you will see that the "face_0-n.csv" file was created. 

        npm install node-fetch fs
        node getUrls.js face 0
        // Code that receives links to download pictures 
        // and stores them on csv

2. **download images**
    - csv : name of the csv file where the links you are trying to download are stored
    - dir : the folder where the pictures you're trying to download will be stored

        python downloader.py --csv face_0-n.csv --dir images

Finally, to make csv file, run this command.

- imgdir : image directory which has downloaded images.
- csvdir : directory for saving csv file. (our csvdir has 2 subdirectory train, val)

    python crawled_data_process.py --imgdir images --csvdir /root/csv/train

### Faceforensics deepfakes dataset

[http://kaldir.vc.in.tum.de/faceforensics_benchmark/documentation](http://kaldir.vc.in.tum.de/faceforensics_benchmark/documentation)

To use faceforensic dataset, you have to fill the google form.

If you run this command, you can download 100 face-forensics deepfakes video and see that the "manipulated_sequences" folder was created.

    python forensic_download.py . -d Deepfakes -n 100

To split video into frame, make images folder to save split image, and you have to run below command.

    python spliter.py _ images True ./manipulated_sequences/Deepfakes/raw/videos

Finally, to make csv file, run this command.

- imgdir : image directory which has split image.
- csvdir : directory for saving csv file. (our csvdir has 2 subdirectory train, val)

    python forensic_data_process.py --imgdir images --csvdir /root/csv

### Deepfake TIMIT

[https://www.idiap.ch/dataset/deepfaketimit](https://www.idiap.ch/dataset/deepfaketimit)

deepfakeTIMIT dataset consists of videos. So, to split video by frame, run this command.

imgdir : image directory for saving split image.

viddir : directory which has deepfakeTIMIT dataset.

    python TIMIT_data_process.py --imgdir /root/data/deepfakeTIMIT --viddir /root/rawData/DeepfakeTIMIT --split True

Finally, to make csv file, run this command.

- imgdir : image directory which has split images.
- csvdir : directory for saving csv file. (our csvdir has 2 subdirectory train, val)

    python TIMIT_data_process.py --imgdir /root/data/deepfakeTIMIT --csvdir /root/csv

### UADFV

[https://drive.google.com/drive/u/0/folders/1GEk1DSxmlV_61JtpEGzC9Fo_BffvyxpH](https://drive.google.com/drive/u/0/folders/1GEk1DSxmlV_61JtpEGzC9Fo_BffvyxpH)

This dataset contains 98 videos, which having 49 real videos and 49 fake videos respectively. 

To split video into frame, create UADFV folder in /root/data and create fake, real folder in it. and to save split image, and you have to run below command.

    python spliter.py _ /root/data/UADFV/fake True /root/rawData/UADFV/fake
    python spliter.py _ /root/data/UADFV/real True /root/rawData/UADFV/real

To make csv file, run this command.

- dir : image directory which has split images.

    python UADFV_data_process.py --dir /root/data/UADFV

### Aberdeen datasets

[http://pics.psych.stir.ac.uk/2D_face_sets.htm](http://pics.psych.stir.ac.uk/2D_face_sets.htm)

Aberdeen datasets are consists of several face image datasets and this is widely used in deep-fake field of deep learning. We utilized [aberdeen](http://pics.psych.stir.ac.uk/zips/Aberdeen.zip) set and [Utrecht ECVP](http://pics.psych.stir.ac.uk/zips/utrecht.zip) which are face images that individuals are looking forward. These datasets consists of only true face image data, no deep-fake images.

After you download zip file, you should unzip the zip files. Then there will be single folder for each dataset zip files.

To make csv file, run this command.

- original_path: image directory which true face images, can get multiple inputs to make one csv file.
- csv_path: directory for saving csv file. (our csvdir has 2 subdirectory train, val)

    python TRUE_data_process.py --original_path /root/data/Aberdeen --csv_path /root/csv/train/Aberdeen.csv

### MUCT face datasets

[https://github.com/StephenMilborrow/muct](https://github.com/StephenMilborrow/muct)

MUCT datasets also consists of several face image datasets and this also widely used in deep-fake field of deep learning. We utilized all of muct dataset exists in that link from muct-a to muct-e datasets.

At first if you download with the link, strange filename added after tar.gz and you should change that filename into tar.gz. After you changed file into tar.gz file, you should unzip tar.gz file via "tar" command in terminal if you are using linux or Mac. Then you'd better to gather all face images into single folder.

To make csv file, run this command.

- original_path: image directory which true face images, can get multiple inputs to make one csv file.
- csv_path: directory for saving csv file. (our csvdir has 2 subdirectory train, val)

    python TRUE_data_process.py --original_path /root/data/muct --csv_path /root/csv/train/muct.csv

## Pre-trained models

> Due to the size of the pre-trained save files, they are uploaded in the following google drive link, not github.

- To download the pre-trained save files,
    - Windows users can download save files from [here](https://drive.google.com/open?id=1MRWAipur5Q6nLA11y31GmDTnkAu2ZaQ9).
    - Linux or Mac users can download in terminal directly with following methods.
        1. From [here](https://drive.google.com/drive/folders/1YrNOprLxHT2Njgvk-EdpxqIsccXOiR0N), select a file that is need to be downloaded and do right click.
        2. Copy the [link](https://drive.google.com/drive/folders/1YrNOprLxHT2Njgvk-EdpxqIsccXOiR0N) for sharing like [https://drive.google.com/open?id=1MRWAipur5Q6nLA11y31GmDTnkAu2ZaQ9](https://drive.google.com/open?id=1MRWAipur5Q6nLA11y31GmDTnkAu2ZaQ9)
        3. Extract file ID like from above 1MRWAipur5Q6nLA11y31GmDTnkAu2ZaQ9
        4. Fill file ID in FILEID and fill download filename in FILENAME in this command:

        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt

- How to use it

    You can use this save file for evaluation of our models. This will be explained more in below Usage - Test section.

# Code Review

## Gaussian Blur

> Some images go through a Gaussian blur process to intentionally generate images with artificial crafts.

- Purpose of this process is to generate fake images with warping artifacts for training. Without collecting fake images or going through other complicated process, fale image can easily generated by using GaussianBlur function. Also, training with this metheod is proved to be effective in the paper.

    def _transform_function(self, img):
          face_cascade = cv2.CascadeClassifier(
              os.getcwd()+'/haarcascade_frontface.xml')  # data provided from open cv
          gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
          faces = face_cascade.detectMultiScale(
              gray, 1.3, 5)  # detecting face from received img
          
       ~~~(omitted)~~~
    
              alimg = cv2.resize(cropimg, dsize=(0, 0), fx=fx,
                                 fy=fy, interpolation=cv2.INTER_LINEAR)
              temp = cv2.GaussianBlur(alimg, (5, 5), 0)  # 얼굴부분 추출해서 Gaussian Blur
              blur = cv2.resize(temp, dsize=(0, 0), fx=1/fx, fy=1/fy,
                                interpolation=cv2.INTER_LINEAR)
    
    	 ~~~(omitted)~~~

1. The face part of the input image is detected with help of haarcascade_frontface.xml, which is a trained data provided from OpenCV library.
2. The face part is cropped out, and randomly resized.
3. The resized face image go through a Gaussian Blur process
4. Face image is the converted back to original face size, placed back again at the original image.

# Usage

## Train

> You can handle some parameters for training.

Since our data processing part don't support high-resolution image,  the recommended input images' width and height should be less than 700px. Following is command for train specific model.

In the command shown at below, you can add arguments before the 'csv' to control the training condition and add arguments after the 'csv' to handle the data that you want to use for training. Since there are too much arguments for control training condition, several necessary arguments will be introduced here.

- Controlling training condition
    - model: Model name, set as resnet50 as default. To add other models for training, you should make model in "deepfake-detection/models" directory.
    - snapshot: The pretrained model's directory used for resume training from the save point of previous experiments.
    - tensorboard-dir: log directory for tensorboard outputs. Set as './logs' as default.
- handling data for training

    The first argument after 'csv' is training csv file directory for training which are usually placed in 'csv/train' directory. Also, argument 'val-annotations' is validation csv file directory used in training which are usually placed in 'csv/val' directry.

    python -u train.py --model vgg16 csv /root/csv/train/train.csv --val-annotations=/root/csv/val/validation.csv >> vgg16.log

## Test

> You can load some pretrained model and check the accuracy for that pretrained model to test thee performance of trained model.

Since our data processing part don't support high-resolution image,  the recommended input images' width and height should be less than 700px. Followings are several arguments that can be changed by training conditions and evaluation condition.

- model: Model name, set as resnet50 as default.
- batch-size: Size of batches used for data processing.
- gpu: Id of the GPU to use as reported by `nvidia-smi`.
- multi-gpu: Number of GPUs to use for parallel processing, integer number.
- multi-gpu-force: Extra flag needed to enable experimental multi-gpu-support
- config: path to a configuration parameters .ini file.

These seven arguments should be added in front of csv in below code and these are set with default values.

The two arguments after csv in below code are evaluation csv data path and pretrained weight at certain epoch, respectively.

    python evaluate.py csv /root/csv/val/validation.csv /root/deepfake-detection/deepfake-detection/bin/snapshots/resnet50_csv_40.h5