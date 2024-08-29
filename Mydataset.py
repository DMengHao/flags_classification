import pickle
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, train: bool):
        if train==True:
            with open('./utils/train_data_cache.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('./utils/val_data_cache.pkl', 'rb') as f:
                self.data = pickle.load(f)
    def __len__(self):
        return len(self.data['labels'])
    def __getitem__(self, idx):
        return self.data['pictures'][idx], self.data['labels'][idx]

if __name__ == '__main__':
    dataset = MyDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    pictures = dataset.data['pictures']
    for i,picture in enumerate(pictures):
        plt.imsave(f"./pictures/{i:8d}.png", picture)
    for i, data in enumerate(dataloader):
        print(i, data)

