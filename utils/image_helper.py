"""
Author: dizhong zhu
Date: 24/09/2022
"""


def make_video(imgs, out_path, fps=10):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(out_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
        video.write(img)
    video.release()
