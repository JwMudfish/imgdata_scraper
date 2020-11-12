import pyzbar.pyzbar as pyzbar
import cv2

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, aliased
from keys import keys
import models
engine = create_engine(f'postgresql://postgres:{keys.get("postgres", "./keys")}@database-1.ctnphj2dxhnf.ap-northeast-2.rds.amazonaws.com/emart24')
Session = sessionmaker(bind=engine)
session = Session()

def get_goods_name(goods_id):
    result = session.query(models.Good.goods_name).filter_by(goods_id=goods_id).first()
    return result.goods_name


cap = cv2.VideoCapture(2)


MJPG_CODEC = 1196444237.0 # MJPG


cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)
cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FOCUS, 100)


i = 0
while(cap.isOpened()):
  ret, img = cap.read()

  if not ret:
    continue

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
  decoded = pyzbar.decode(gray)

  for d in decoded: 
    x, y, w, h = d.rect

    barcode_data = d.data.decode("utf-8")
    barcode_type = d.type

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    text = '%s (%s)' % (barcode_data, barcode_type)
    goods_name = barcode_data
    #cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, goods_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

  cv2.imshow('img', img)

  key = cv2.waitKey(1)
  if key == ord('q'):
    break
  elif key == ord('s'):
    i += 1
    cv2.imwrite('c_%03d.jpg' % i, img)

cap.release()
cv2.destroyAllWindows()