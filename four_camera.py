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
#import xml.etree.ElementTree as ET
#from xml.dom import minidom
import os
import random
import pyzbar.pyzbar as pyzbar

# sudo apt install v4l-utils
#################################
BRIGHTNESS = 10
save_dir = './saved_images'

CAMERA_NUM_1 = 0
CAMERA_NUM_2 = 2
CAMERA_NUM_3 = 4
CAMERA_NUM_4 = 6


frame_width = int(1920)
frame_height = int(1080)

###################################

MODE = 'b'   # b : box 모드
BOX_NUM_1 = '0'   # 
BOX_NUM_2 = '0'

def make_folder(label_dir):
    if not os.path.exists(save_dir +'/' + label_dir):
        os.makedirs(save_dir +'/' + label_dir)

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
frame0.set(cv2.CAP_PROP_FOCUS, 0)

frame1.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS) 
frame1.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame1.set(cv2.CAP_PROP_AUTOFOCUS, 1)
#frame1.set(cv2.CAP_PROP_FOCUS, 1)


frame2.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS) 
frame2.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame2.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame2.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
frame2.set(cv2.CAP_PROP_FOCUS, 0)


frame3.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS) 
frame3.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame3.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame3.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame3.set(cv2.CAP_PROP_AUTOFOCUS, 0)
frame3.set(cv2.CAP_PROP_FOCUS, 0)


RESIZE = (1920,1080)
cv2.createTrackbar('Brightness','Interminds Train Image Collection Program by JW', 100, 200, nothing)


while True:

    frame0.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    frame1.set(cv2.CAP_PROP_BRIGHTNESS, -50)
    frame2.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    frame3.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)


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

    cv2.putText(final_rst, 'Cam3_1', (50, 1200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(final_rst, 'Cam1_3', (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    cv2.putText(final_rst, 'Cam4_2', (4150, 1200), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(final_rst, 'Cam2_4', (4150, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    BRIGHTNESS = cv2.getTrackbarPos('Brightness', 'Interminds Train Image Collection Program by JW') - 100


    cv2.imshow('Interminds Train Image Collection Program by JW',final_rst)
    today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')


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
        cv2.imwrite(f'./{save_dir}/cam1({today}).jpg', image1)
        cv2.imwrite(f'./{save_dir}/cam2({today}).jpg', image2)
        cv2.imwrite(f'./{save_dir}/cam3({today}).jpg', image3)
        cv2.imwrite(f'./{save_dir}/cam4({today}).jpg', image4)

        print(f'./{save_dir}/cam1234({today}).jpg')

        #crop_image(image, box, save_dir,  LABELS, (RESIZE,RESIZE))
        
    elif ch == ord('b'):
        print('press B')
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
     
        decoded = pyzbar.decode(gray)

        for d in decoded: 
            x, y, w, h = d.rect

            barcode_data = d.data.decode("utf-8")
            barcode_type = d.type

            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            text = '%s' % (barcode_data)
            try:
                print(text)
                #cv2.putText(img, barcode_data, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            except:
                print('------------------------------')


frame0.release()
frame1.release()
frame2.release()
frame3.release()
cv2.destroyAllWindows()