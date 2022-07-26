import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'left': 100, 'top': 0, 'width': 900, 'height': 1000}

with mss() as sct:
    x = 0
    y = 0
    w = 0
    h = 0
    while True:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGBA',
            (screenShot.width, screenShot.height),
            screenShot.bgra,
        )
        # cv2.imshow('test', np.array(img))
        img = np.array(img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold of blue in HSV space
        lower_skin = np.array([2, 75, 255])
        upper_skin = np.array([22, 95, 255])

        mask = cv2.inRange(hsv, lower_skin, upper_skin)


        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        print('---------------')
        box = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            print(x, y, w, h)
        cv2.rectangle(box, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # ROI = img[y:y + h, x:x + w]
            # cv2.imwrite('Image_{}.png'.format(ROI_number), ROI)
            # ROI_number += 1

        cv2.imshow('mask', box)

        if cv2.waitKey(33) & 0xFF in (
                ord('q'),
                27,
        ):
            break
