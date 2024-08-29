import json
import os
import pickle
from itertools import chain
import cv2
import numpy as np
from matplotlib import pyplot as plt
import re

class preprocess():
    def __init__(self):
        pass

    @classmethod
    def read_frame_file(cls, video_path):
        frame_list = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame {}'.format(i))
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        cap.release()
        return frame_list

    @classmethod
    def read_joint_file(cls, Joint_file_path):
        Joint_list = []
        joints = []
        with open(Joint_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 0:
                    joints = joints
                if len(parts) == 3:
                    joint_tuple = list(map(float, parts))
                    joints.append(joint_tuple)
                if "no_person" in line:
                    joint_tuple = ()
                    for i in range(1, 27):
                        joints.append(joint_tuple)
        Joints_lenght = len(joints)
        frame_length = int(Joints_lenght / 26)
        j = 0
        temp_tuple = []
        for i in range(0, frame_length):
            while j % 26 != 0 or j == 0:
                temp_tuple.append(joints[j])
                j += 1
            Joint_list.append(temp_tuple)
            if j == Joints_lenght:
                break
            temp_tuple = []
            temp_tuple.append(joints[j])
            j = j + 1
        Joint_list = [[j[:2] for j in i] for i in Joint_list]
        return Joint_list

    @classmethod
    def get_left_handed_flag(cls,joint, picture):
        x1 = joint[7]
        x2 = joint[9]
        direction = [b - a for a, b in zip(x1, x2)]
        direction_unit = direction / np.linalg.norm(direction)
        width = 100
        height = 100
        # Unit vectors perpendicular to direction vectors
        perpendicular_unit = np.array([direction_unit[1], -direction_unit[0]]) / np.linalg.norm(direction_unit)
        # x2 = x2 + direction_unit*60
        x2 = x2 +direction_unit*60
        # x2 is the center point to compute the four vertices of the crop
        top_left = x2 + direction_unit * width / 2 - perpendicular_unit * height / 2
        top_right = x2 + direction_unit * width / 2 + perpendicular_unit * height / 2
        bottom_left = x2 - direction_unit * width / 2 - perpendicular_unit * height / 2
        bottom_right = x2 - direction_unit * width / 2 + perpendicular_unit * height / 2
        pts1 = np.float32([top_left, top_right, bottom_left])
        pts2 = np.float32([[0, 0], [width, 0], [0, height]])
        # Compute the affine transformation matrix
        M = cv2.getAffineTransform(pts1, pts2)
        cropped_image = cv2.warpAffine(picture, M, (width, height))
        cropped_image = cv2.resize(cropped_image,(50,50))
        # cropped_image = cv2.GaussianBlur(cropped_image, ksize=(3, 3), sigmaX=1, sigmaY=1)
        return cropped_image/255.0

    @classmethod
    def get_right_handed_flag(cls, joint, picture):
        x1 = joint[8]
        x2 = joint[10]
        direction = [b - a for a, b in zip(x1, x2)]
        direction_unit = direction / np.linalg.norm(direction)
        width = 100
        height = 100
        perpendicular_unit = np.array([-direction_unit[1], -direction_unit[0]]) / np.linalg.norm(direction_unit)
        # x2 = x2 + direction_unit*60
        x2 = x2 + direction_unit*60
        top_left = x2 + direction_unit * width / 2 - perpendicular_unit * height / 2
        top_right = x2 + direction_unit * width / 2 + perpendicular_unit * height / 2
        bottom_left = x2 - direction_unit * width / 2 - perpendicular_unit * height / 2
        bottom_right = x2 - direction_unit * width / 2 + perpendicular_unit * height / 2
        pts1 = np.float32([top_left, top_right, bottom_left])
        pts2 = np.float32([[0, 0], [width, 0], [0, height]])
        M = cv2.getAffineTransform(pts1, pts2)
        cropped_image = cv2.warpAffine(picture, M, (width, height))
        cropped_image = cv2.resize(cropped_image,(50,50))
        # cropped_image = cv2.GaussianBlur(cropped_image, ksize=(3, 3), sigmaX=1, sigmaY=1)
        return cropped_image/255.0

    @classmethod
    def get_one_joint_picture(cls, video_path, joint_path):
        with open(f"E:/recovery_source_code/flags_classification/utils/class_indices.json", 'r', encoding='utf-8') as file:
            class_indices = json.load(file)
        class_ = class_indices[os.path.basename(video_path).split('_')[0]]  if os.path.basename(video_path).split('_')[0] in class_indices else None
        data = {}
        data['labels'] = []
        data['pictures'] = []
        joints = preprocess.read_joint_file(joint_path)
        selected_joints = [joints[i:i+2] for i in range(0, len(joints), 240)]
        joints = list(chain.from_iterable(selected_joints))
        pictures = preprocess.read_frame_file(video_path)
        selected_pictures = [pictures[j:j+2] for j in range(0, len(pictures), 240)]
        pictures = list(chain.from_iterable(selected_pictures))
        for i, joint in enumerate(joints):
            temp1 = preprocess.get_left_handed_flag(joint, pictures[i])
            data['pictures'].append(temp1)
            data['labels'].append(class_[0])
            temp2 = preprocess.get_right_handed_flag(joint, pictures[i])
            data['pictures'].append(temp2)
            data['labels'].append(class_[1])
        return data

    @classmethod
    def get_all_pictures_joints_lables(cls,videos_path,joints_path):
        data = {}
        data['pictures'] = []
        data['labels'] = []
        joints = os.listdir(joints_path)
        i =0
        for filename in os.listdir(videos_path):
            file_path = os.path.join(videos_path, filename)
            joint_path = os.path.join(joints_path, joints[i])
            temp = preprocess.get_one_joint_picture(file_path, joint_path)
            data['pictures'] = data['pictures'] + temp['pictures']
            data['labels'] = data['labels'] + temp['labels']
            i +=1
        return data

    @classmethod
    def delete_data(self):
        print("########This is third get all data, and save pkl file.########")
        train_picture_path = '../Pictures/train'
        save_list_number = []
        for filename in os.listdir(train_picture_path):
            save_list_number.append(int(re.search(r'(\d+)',filename).group(0)))
        with open('train_data_cache.pkl', 'rb') as f:
            train = pickle.load(f)
        train_data = {}
        train_pictures = []
        train_labels = []
        for i, _ in enumerate(train['pictures']):
            if i in save_list_number:
                train_pictures.append(train['pictures'][i])
                train_labels.append(train['labels'][i])
        train_data['pictures'] = train_pictures
        train_data['labels'] = train_labels
        with open('train_data_cache.pkl', 'wb') as f:
            pickle.dump(train_data, f)

        val_picture_path = '../Pictures/val'
        save_list_number_ = []
        for filename in os.listdir(val_picture_path):
            save_list_number_.append(int(re.search(r'(\d+)', filename).group(0)))
        with open('val_data_cache.pkl', 'rb') as f:
            val = pickle.load(f)
        val_data = {}
        val_pictures = []
        val_labels = []
        for i_, _ in enumerate(val['pictures']):
            if i_ in save_list_number:
                val_pictures.append(val['pictures'][i_])
                val_labels.append(val['labels'][i_])
        val_data['pictures'] = val_pictures
        val_data['labels'] = val_labels
        with open('val_data_cache.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        print("train dataset length ","pictures:",len(train_data['pictures']),"labels:",len(train_data['labels']))
        print("val dataset length:","pictures:",len(val_data['pictures']),"labels:",len(val_data['labels']))

if __name__ == '__main__':
    first_save_data = True
    third_save_data = False

    if first_save_data:
        print("#######This is first get all data, and save pkl file.########")
        train_data = preprocess.get_all_pictures_joints_lables("../Datasets/train/videos",'../Datasets/train/joints')
        val_data = preprocess.get_all_pictures_joints_lables('../Datasets/val/videos','../Datasets/val/joints')
        print("########### Get all data None! ##########")

        with open('train_data_cache.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        print(f"cache file saved at train_data_cache.pkl")
        with open('train_data_cache.pkl', 'rb') as f:
            test = pickle.load(f)
        print(len(test['pictures']),'####',len(test['labels']))
        # for i,picture in enumerate(test['pictures']):
        #     plt.imsave(f"../pictures/train/{i}.png", picture)

        with open('val_data_cache.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        print(f"cache file saved at val_data_cache.pkl")
        with open('val_data_cache.pkl', 'rb') as f:
            test_ = pickle.load(f)
        print(len(test_['pictures']),'####',len(test_['labels']))
        # for i_,picture_ in enumerate(test_['pictures']):
        #     plt.imsave(f"../pictures/val/{i_}.png", picture_)
        print("########## Preprocessing Done! ###########")
    # if third_save_data:
    #     preprocess.delete_data()
