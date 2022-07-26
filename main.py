import random

import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller
import time

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


def generate_actions():
    actions = []
    for horizontal in [Key.left, Key.right, None]:
        for vertical in [Key.up, Key.down, None]:
            for key in ['z', 'x', None]:  # Jump, dash, none
                if key != 'x' and vertical == Key.down:
                    continue
                if key is None and vertical is not None:
                    continue

                actions.append([horizontal, vertical, key])
    return actions


def reset_level(keyboard: Controller):
    keyboard.press('f')
    time.sleep(0.2)
    keyboard.release('f')
    time.sleep(0.2)
    keyboard.press('s')
    time.sleep(0.2)
    keyboard.release('s')
    time.sleep(0.4)


keyboard = Controller()

best_score = -999999
last_press = time.time()
start = time.time()
actions = generate_actions()

if __name__ == '__main__':
    with mss() as sct:
        while True:
            x_pos, y_pos, _, _ = get_position(sct)
            score = -y_pos
            if score > best_score:
                best_score = score
            print(best_score)
            action = random.choice(actions)

            if time.time() - last_press > 0.01:
                for key in action:
                    if key is not None:
                        keyboard.press(key)
                time.sleep(0.1)
                for key in action:
                    if key is not None:
                        keyboard.release(key)
                last_press = time.time()

            if time.time() - start > 5:
                reset_level(keyboard)
                start = time.time()
