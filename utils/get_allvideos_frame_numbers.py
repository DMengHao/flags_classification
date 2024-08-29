import os

import cv2
from tqdm import tqdm

path = '../Datasets/train/videos'
filenames_list = os.listdir(path)
numbers = 0
for i in tqdm(range(len(filenames_list))):
    file_path = os.path.join(path, filenames_list[i])
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numbers += frame_count
    cap.release()
print(f'path:../Dataset/train/videos    all_frames:{numbers}')
print(f'path:../Dataset/train/videos    split_image:{numbers*2}')