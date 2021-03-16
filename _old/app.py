#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

import pyautogui

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.75)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.75)

    parser.add_argument('--use_left_hand', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_left_hand = args.use_left_hand

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        upper_body_only=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    #  ########################################################################
    # 人差指のID
    ID_FINGER_TIP = 8

    history_length = 6
    point_x_history = deque(maxlen=history_length)
    point_y_history = deque(maxlen=history_length)
    point_z_history = deque(maxlen=history_length)

    display_size = pyautogui.size()

    # FPS計測モジュール ########################################################
    start_time = time.time()
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        image_width, image_height = image.shape[1], image.shape[0]
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        # Hands ###############################################################
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        landmarks = None
        # 右手
        if (left_hand_landmarks is not None) and (not use_left_hand):
            landmarks = calc_hands_landmarks(image, left_hand_landmarks)
        # 左手
        if (right_hand_landmarks is not None) and (use_left_hand):
            landmarks = calc_hands_landmarks(image, right_hand_landmarks)

        if landmarks is not None:
            point_x_history.append(landmarks[ID_FINGER_TIP][0])
            point_y_history.append(landmarks[ID_FINGER_TIP][1])
            point_z_history.append(landmarks[ID_FINGER_TIP][2])

            point_x = int(sum(point_x_history) / len(point_x_history))
            point_y = int(sum(point_y_history) / len(point_y_history))
            point_z = point_z_history[-1]

            # 描画
            debug_image = draw_hands_landmarks(debug_image, point_x, point_y,
                                               point_z, display_fps)

            mouse_x = int(display_size.width * (point_x / image_width))
            mouse_y = int(display_size.height * (point_y / image_height))

            if (time.time() - start_time) > 0.3:
                start_time = time.time()
                pyautogui.moveTo(mouse_x, mouse_y)

            if len(point_z_history) >= history_length:
                diff_z = max(point_z_history) - min(point_z_history)
            else:
                diff_z = 0
            max_index_z = point_z_history.index(max(point_z_history))
            min_index_z = point_z_history.index(min(point_z_history))
            if diff_z > 0.10 and max_index_z < min_index_z:
                pyautogui.click()
                point_z_history.clear()

        else:
            if len(point_z_history):
                point_z_history.popleft()

        # 画面反映 #############################################################
        image_width, image_height = debug_image.shape[1], debug_image.shape[0]
        debug_image = cv.resize(debug_image,
                                (int(image_width / 2), int(image_height / 2)))
        cv.imshow('Simple Virtual Mouse Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_hands_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y, landmark_z))

    return landmark_point


def draw_hands_landmarks(
    image,
    point_x,
    point_y,
    point_z,
    display_fps,
):
    cv.circle(image, (point_x, point_y), 12, (255, 255, 255), 8)
    cv.circle(image, (point_x, point_y), 12, (0, 0, 0), 2)

    cv.putText(image, "z:" + str(round(point_z, 3)),
               (point_x - 20, point_y - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
               (255, 255, 255), 6, cv.LINE_AA)
    cv.putText(image, "z:" + str(round(point_z, 3)),
               (point_x - 20, point_y - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
               (0, 0, 0), 2, cv.LINE_AA)

    cv.putText(image, "FPS:" + str(display_fps), (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(display_fps), (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
