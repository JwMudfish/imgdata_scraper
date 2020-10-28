import pandas as pd
import cv2
import numpy as np
import imutils
from datetime import datetime
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os


BOX_NUM = '0'   # 
MODE = 'b'   # b : box 모드

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

def get_labels(label_path):
    with open('./label.txt', 'r') as file:
        labels = file.readlines()
        labels = list(map(lambda x : x.strip(), labels))

    return labels

frame0 = cv2.VideoCapture(0)
frame1 = cv2.VideoCapture(1)

frame_width = int(1920)
frame_height = int(1080)

MJPG_CODEC = 1196444237.0 # MJPG
BRIGHTNESS = 10

cv2.namedWindow('Usb Cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Usb Cam', frame_width,frame_height)

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


while True:

    ret0, img0 = frame0.read()
    ret1, img00 = frame1.read()
    
    box = get_boxes('./boxes')['{}'.format(BOX_NUM)]
    box_name = sorted(os.listdir('./boxes'))
    LABELS = get_labels('./label.txt')

    # if MODE =='b':
    #     for i, j in zip(box, LABELS):
    #         cv2.rectangle(img0, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)
    #         cv2.putText(img0, j, (i[0], i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            
    #         cv2.rectangle(img00, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)
    #         cv2.putText(img00, j, (i[0], i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
        
        
        #cv2.putText(img0, 'BOX : {}'.format(BOX_NUM), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    if MODE =='b':
        textSize1 = textSize2 = -60
        
        for i, j in zip(box, LABELS):
            cv2.rectangle(img0, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)
            #cv2.putText(frame, j, (i[0], i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)

            cv2.putText(img0, '{} /'.format(j), (textSize1 + 70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            label_len = cv2.getTextSize(text=str(j+'//'), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)[0][0]
            textSize1 = label_len + textSize1
    
            cv2.rectangle(img00, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)
            #cv2.putText(frame, j, (i[0], i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)

            cv2.putText(img00, '{} /'.format(j), (textSize2 + 70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            label_len = cv2.getTextSize(text=str(j+'//'), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)[0][0]
            textSize2 = label_len + textSize2

        cv2.putText(img0, 'BOX : {}_{}'.format(BOX_NUM, box_name[int(BOX_NUM)]), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 3)
        cv2.putText(img00, 'BOX : {}_{}'.format(BOX_NUM, box_name[int(BOX_NUM)]), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 3)


    img1 = cv2.resize(img0,(1920,1080))
    img2 = cv2.resize(img00,(1920,1080))


    rst = cv2.hconcat([img1, img2])
    cv2.imshow('Usb Cam', rst)
   
    ch = cv2.waitKey(1)

    # 종료
    if ch == ord('q'):
	    break

    # 박스 숨기기 / 나타내기
    elif ch == ord('a'):
        if MODE == 'b':
            MODE = 'a'
        else:
            MODE = 'b'


frame0.release()
frame1.release()
cv2.destroyAllWindows()