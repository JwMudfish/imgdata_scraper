#-*- coding: utf-8 -*-
'''
coding by JW_Mudfish
Version : 2.2 
Last Updated 2020.11.12
'''

import pandas as pd
import cv2
import numpy as np
import imutils
from datetime import datetime
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import random

# sudo apt install v4l-utils
###############################
BRIGHTNESS = 10
save_dir = './cls_seed_images'

CAMERA_NUM_1 = 0
CAMERA_NUM_2 = 2
CAMERA_NUM_3 = 4
CAMERA_NUM_4 = 6


frame_width = int(1920)
frame_height = int(1080)

################################

MODE = 'b'   # b : box 모드
BOX_NUM_1 = '0'   # 
BOX_NUM_2 = '0'

def get_boxes(label_path):
    label_path = label_path
    xml_list = os.listdir(label_path)

    boxes_1 = {}
    cnt = 0
    for xml_file in sorted(xml_list):
        if xml_file =='.DS_Store':
            pass
        else:
                #try:
            xml_path = os.path.join(label_path,xml_file)

            root_1 = minidom.parse(xml_path)
            bnd_1 = root_1.getElementsByTagName('bndbox')

            result = []
            for i in range(len(bnd_1)):
                xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
                ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
                xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
                ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)
                result.append((xmin,ymin,xmax,ymax))

            boxes_1[str(cnt)] = result
            cnt += 1
    
    return boxes_1

def make_folder(label_dir):
    if not os.path.exists(save_dir +'/' + label_dir):
        os.makedirs(save_dir +'/' + label_dir)

def get_labels(label_path):
    with open(f'{label_path}', 'r') as file:
        labels = file.readlines()
        labels = list(map(lambda x : x.strip(), labels))

    return labels

def get_multi_labels(label_path):
    with open(f'{label_path}', 'r') as file:
        labels = file.readlines()
        labels = list(map(lambda x : x.strip(), labels))
    
    left = labels[:int(np.ceil(len(labels) / 2))]
    right = labels[int(np.ceil(len(labels) / 2)) :]
    return left, right


def crop_random_image(image, boxes, save_path, labels, left_right, resize = None):
    seed_image = image

    images_1 = list(map(lambda b : image[b[1]:b[3], b[0]:b[2]], boxes))
    images_2 = list(map(lambda b : image[b[1]+random.randint(100,170) : b[3], b[0] : b[2]], boxes))
    images_3 = list(map(lambda b : image[b[1] : b[3], b[0]+random.randint(10,50) : b[2]], boxes))
    images_4 = list(map(lambda b : image[b[1]+random.randint(100,170) : b[3], b[0]+random.randint(10,50) : b[2]], boxes))

    image_list = [images_1, images_2, images_3, images_4]
    num = 0
    
    for images in image_list:
        for img, label in zip(images, labels):
            num = num + 1
            cv2.imwrite('{}/{}/{}_{}_{}_{}.jpg'.format(save_path,label,today,label,left_right, num), img)

    print('Random crop image 함수실행!!!')

def file_count(save_path):
    dir_count = sorted(os.listdir(save_path))
    
    result = []
    for i in dir_count:
        folder_name = i
        files = len(os.listdir(f'{save_path}/{i}'))
        result.append([folder_name, files])
    return result

def getDevicesList():
    devices_list = []

    result = os.popen('v4l2-ctl --list-devices').read()
    result_lists = result.split("\n\n")
    for result_list in result_lists:
        if result_list != '':
            result_list_2 = result_list.split('\n\t')
            devices_list.append(result_list_2[1][-1])
    return devices_list

def nothing(x):
    pass


active_cam = getDevicesList()


print(f'현재 활성화 되어있는 카메라는 {active_cam} 입니다.')

frame0 = cv2.VideoCapture(CAMERA_NUM_1)
frame1 = cv2.VideoCapture(CAMERA_NUM_2)
frame2 = cv2.VideoCapture(CAMERA_NUM_3)
frame3 = cv2.VideoCapture(CAMERA_NUM_4)

MJPG_CODEC = 1196444237.0 # MJPG


cv2.namedWindow('Interminds Train Image Collection Program by JW', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Interminds Train Image Collection Program by JW', frame_width,frame_height)

frame0.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
frame0.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame0.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame0.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame0.set(cv2.CAP_PROP_AUTOFOCUS, 0)

frame1.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS) 
frame1.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

frame2.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS) 
frame2.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame2.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame2.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame2.set(cv2.CAP_PROP_AUTOFOCUS, 0)

frame3.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS) 
frame3.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame3.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame3.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame3.set(cv2.CAP_PROP_AUTOFOCUS, 0)

RESIZE = (1920,1080)

while True:

    frame0.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    frame1.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)

    ret0, img0 = frame0.read()
    ret1, img00 = frame1.read()
    ret2, img000 = frame2.read()
    ret3, img0000 = frame3.read()

    img1 = cv2.resize(img0,RESIZE)
    img2 = cv2.resize(img00,RESIZE)
    img3 = cv2.resize(img000,RESIZE)
    img4 = cv2.resize(img0000,RESIZE)

    rst1 = cv2.hconcat([img1, img2])
    rst2 = cv2.hconcat([img3, img4])
    

    merged_rst = cv2.vconcat([rst1, rst2])

    left_blank_image = np.zeros((merged_rst.shape[0], 300, 3), np.uint8)
    
    #cv2.putText(left_blank_image, 'Cam_3', (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    #cv2.putText(left_blank_image, 'Cam_1', (50, 1200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    right_blank_image = np.zeros((merged_rst.shape[0], 300, 3), np.uint8)
    # cv2.putText(right_blank_image, 'Cam_4', (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    # cv2.putText(right_blank_image, 'Cam_2', (50, 1200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)


    final_rst_1 =  cv2.hconcat([left_blank_image, merged_rst])
    final_rst = cv2.hconcat([final_rst_1, left_blank_image])

    cv2.putText(final_rst, 'Cam_1', (50, 1200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(final_rst, 'Cam_3', (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    cv2.putText(final_rst, 'Cam_2', (4150, 1200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(final_rst, 'Cam_4', (4150, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow('Interminds Train Image Collection Program by JW',final_rst)

    ch = cv2.waitKey(1)

######################################################################################################################
    # 종료
    if ch == ord('q'):
	    break

    elif ch == ord('d'):
        
        image1 = img1
        image2 = img2
        image3 = img3
        image4 = img4
        
        print('press c')
	    cv2.imwrite('./saved_images/usbcam({}).jpg'.format(today),frame)
	    print('./saved_images/usbcam({}).jpg saved'.format(today))

        #crop_image(image, box, save_dir,  LABELS, (RESIZE,RESIZE))
        crop_random_image(image1, box1, save_dir,  LABELS_left, left_right='left')
        crop_random_image(image2, box2, save_dir,  LABELS_right, left_right='right')
        print('{} // {} : left, right section cropped'.format(LABELS_left, LABELS_right))



frame0.release()
frame1.release()
frame2.release()
frame3.release()
cv2.destroyAllWindows()