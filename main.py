import math
import random

import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller
import time
from population_manager import Population
from tqdm import tqdm
POPULATION_SIZE = 20
RUN_TIME = 3.3


def calibrate_screen(sct):
    while True:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGBA',
            (screenShot.width, screenShot.height),
            screenShot.bgra,
        )
        img = np.array(img)
        cv2.imshow('calibrate', img)
        key = cv2.waitKey(1)
        if key == 27 or key == 113:
            cv2.destroyAllWindows()
            return
            break


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
    time.sleep(0.75)

mon = {'left': 200, 'top': 225, 'width': 800, 'height': 800}
x_pos, y_pos = 0, 0

keyboard = Controller()

best_score = -999999
generation_counter = 0
last_press = time.time()
start = time.time()
actions = generate_actions()
pop = Population(population_size=POPULATION_SIZE, agent_num_choices=len(actions), agent_num_actions=int(11*RUN_TIME+0.5))
pop.agents[0].mutation_chance=1


if __name__ == '__main__':
    time.sleep(1)
    with mss() as sct:
        calibrate_screen(sct)

        while True:
            print('------')
            print('Generation:', generation_counter)
            generation_counter += 1

            for agent in tqdm(pop.get_agents()):
                x_pos = 0
                y_pos = 0
                reset_level(keyboard)
                start = time.time()
                for a in agent.get_actions():
                    action = actions[a]
                    x_pos, y_pos, _, _ = get_position(sct)


                    if time.time() - last_press > 0.01:
                        for key in action:
                            if key is not None:
                                keyboard.press(key)
                        time.sleep(0.25)
                        for key in action:
                            if key is not None:
                                keyboard.release(key)
                        last_press = time.time()

                    if time.time() - start > RUN_TIME:
                        break

                score = -math.dist((x_pos, y_pos), (710, 390))
                agent.set_fitness(score)
                if score > best_score:
                    print(f"Improved from {best_score} to {score}")
                    best_score = score
            print("Best Score:", best_score)
            pop.evolve()