#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import pandas as pd


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))
        # c=0
        for i in tqdm(range(nb_frames)):
            # print(i)
            # c+=1
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            image, ratio = draw_boxes(image, boxes, config['model']['labels'])
            # print('ratio: ', ratio)
            # if(c%25==0):
            #     cv2.imwrite(image_path[:-4]+str(i) + '_detected' + '.jpg', image)
            video_writer.write(np.uint8(image))
            # c+=1
        # print(c)    
        video_reader.release()
        video_writer.release()  
    else:
        folder = image_path
        lst = []
        print(folder, os.listdir(folder))
        for path in os.listdir(folder):
            # print("-> ", path)
            image_path = folder + path
            image = cv2.imread(image_path)
            boxes = yolo.predict(image)
            image, ratio = draw_boxes(image, boxes, config['model']['labels'])
            
            
            # print(boxes)
            # print(len(boxes), 'boxes are found')
            
        
            x=len(boxes)
            if x>10 or ratio>0.87:
            	lst.append([image_path, 5])
            elif (x>7 and x<=10) or ratio>0.7:
                lst.append([image_path, 4])
            elif x>4 and x<=7:
            	lst.append([image_path, 3])
            elif x>2 and x<=4:
            	lst.append([image_path, 2])
            else:
            	lst.append([image_path, 1])
    
            cv2.imwrite("mapped_images/" + path, image)
        df = pd.DataFrame(lst, columns=['Images', 'Level'])
        df.to_csv('images_to_level.csv')
        print(lst)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
