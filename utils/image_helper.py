"""
Author: dizhong zhu
Date: 24/09/2022
"""
import cv2
import numpy as np


def make_video(imgs, out_path, fps=10):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(out_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
        video.write(img)
    video.release()


def images_f_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    images = []

    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret:
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break

    return np.stack(images)
