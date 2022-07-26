import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'left': 100, 'top': 0, 'width': 900, 'height': 1000}

x_pos, y_pos = 0, 0
def get_position(sct):
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
    box = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    x, y = x_pos, y_pos
    w, h = 0, 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(box, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # cv2.imshow('mask', box)
    return x, y, h, w

best_score = -999999
if __name__=='__main__':
    with mss() as sct:
        while True:
            x_pos, y_pos, _, _ = get_position(sct)
            score = -y_pos
            if score > best_score:
                best_score = score
            print(best_score)
