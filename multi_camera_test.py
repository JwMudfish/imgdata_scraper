import pandas as pd
import cv2
import numpy as np
import imutils
from datetime import datetime
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os


BOX_NUM_1 = '0'   # 
BOX_NUM_2 = '0'
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
    with open(f'{label_path}', 'r') as file:
        labels = file.readlines()
        labels = list(map(lambda x : x.strip(), labels))

    return labels

def crop_random_image(image, boxes, save_path, labels, resize = None):
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
            cv2.imwrite('{}/{}/{}_{}_{}.jpg'.format(save_path,label,today,label, num), img)

    print('Random crop image 함수실행!!!')


frame0 = cv2.VideoCapture(0)
frame1 = cv2.VideoCapture(2)

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
    
    box1 = get_boxes('./boxes')['{}'.format(BOX_NUM_1)]
    box2 = get_boxes('./boxes')['{}'.format(BOX_NUM_2)]

    box_name = sorted(os.listdir('./boxes'))
    
    LABELS_left = get_labels('./label_left.txt')
    LABELS_right = get_labels('./label_right.txt')

    blank_image_1 = np.zeros((150, frame_width, 3), np.uint8)
    blank_image_2 = np.zeros((150, frame_width, 3), np.uint8)

    if MODE =='b':
        textSize1 = textSize2 = -60
        
        for i in box1:
            cv2.rectangle(img0, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)
    
        for i in box2:
            cv2.rectangle(img00, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)


    textSize1 = textSize2 = -60
    for j in LABELS_left:
        cv2.putText(blank_image_1, '{} /'.format(j), (textSize1 + 70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        label_len = cv2.getTextSize(text=str(j+'//'), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)[0][0]
        textSize1 = label_len + textSize1

    for j in LABELS_right:
        cv2.putText(blank_image_2, '{} /'.format(j), (textSize2 + 70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        label_len = cv2.getTextSize(text=str(j+'//'), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)[0][0]
        textSize2 = label_len + textSize2

    cv2.putText(blank_image_1, 'BOX : {}_{}'.format(BOX_NUM_1, box_name[int(BOX_NUM_1)]), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 3)
    cv2.putText(blank_image_2, 'BOX : {}_{}'.format(BOX_NUM_2, box_name[int(BOX_NUM_2)]), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 3)

    img1 = cv2.resize(img0,(1920,1080))
    img2 = cv2.resize(img00,(1920,1080))

    img1 = cv2.vconcat([blank_image_1, img1])
    img2 = cv2.vconcat([blank_image_2, img2])

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


    # box 변경 ( '[' : - 변경,   ']' : + 변경)
    elif ch == ord('o'):
        BOX_NUM_1 = str(int(BOX_NUM_1) + 1)
    
        if BOX_NUM_1 == str(len(os.listdir('./boxes'))):
            BOX_NUM_1 = '0'

    elif ch == ord('p'):
        BOX_NUM_1 = str(int(BOX_NUM_1) - 1)
    
        if BOX_NUM_1 == '-1':
            BOX_NUM_1 = str(len(os.listdir('./boxes'))-1)

    elif ch == ord('['):
        BOX_NUM_2 = str(int(BOX_NUM_2) + 1)
    
        if BOX_NUM_2 == str(len(os.listdir('./boxes'))):
            BOX_NUM_2 = '0'

    elif ch == ord(']'):
        BOX_NUM_2 = str(int(BOX_NUM_2) - 1)
    
        if BOX_NUM_2 == '-1':
            BOX_NUM_2 = str(len(os.listdir('./boxes'))-1)


frame0.release()
frame1.release()
cv2.destroyAllWindows()