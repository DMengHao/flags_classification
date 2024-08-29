import os
import cv2
from tqdm import tqdm

video_path = "../Temp/videos/指挥机车向显示人反方向去的信号_吴山山1.mp4"
frame_save_path = "../Temp/frame_pictures"

if not os.path.exists(frame_save_path):
    os.makedirs(frame_save_path)
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(frame_count):
    ret, frame = cap.read()
    frame_save_path_temp = os.path.join(frame_save_path, f"{i+1:7d}" + ".png")
    cv2.imwrite(frame_save_path_temp, frame)
cap.release()

