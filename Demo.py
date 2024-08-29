import os.path
import pickle
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils.Resnet18_SAM import ResNet18_SAM as Model
L = ['open_green', 'close_red', 'close_green', 'close_yellow']

train = True
if train:
    data_cache = './utils/train_data_cache.pkl'
    error_picture_path = './Result/Demo/train'
else:
    data_cache = './utils/val_data_cache.pkl'
    error_picture_path = './Result/Demo/val'

def export_onnx():
    model = Model()
    model.eval()
    model.load_state_dict(torch.load('./model_pth/50.0_12.pth'))
    with open(data_cache, 'rb') as f:
        temp = pickle.load(f)
    data = temp['pictures']
    label = temp['labels']
    for i in tqdm(range(len(data))):
        result = model(torch.tensor(data[i]).unsqueeze(0).float())
        result = torch.argmax(result, dim=1)
        if result != label[i]:
            error_picture = data[i]
            fig, ax = plt.subplots()
            ax.imshow(error_picture)
            plt.savefig(os.path.join(error_picture_path, f'{L[label[i]]}_{i}.png'))
            plt.close()
if __name__ == '__main__':
    export_onnx()