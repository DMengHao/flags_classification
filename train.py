import os
import time
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch
from Mydataset import MyDataset
from utils.Resnet18_SAM import ResNet18_SAM as Model
from utils.Confusion_maxtrix import confusion_matrix as ConfusionMatrix
from utils.draw_loss_curve import Draw_loss_curve

epochs = 100

def train():
    '''
        This is train data set.
    '''
    run_time = time.strftime("%Y_%m_%d_%H_%M")
    print("#####################Start load train and val data set.#########################")
    train_dataset = MyDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = MyDataset(train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print("#####################Finish load train and val data set.#########################")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    Mean_loss = []
    for epoch in range(1, epochs):
        print("###########"+'Epoch {}/{}'.format(epoch, epochs)+"##########")
        Loss = 0
        confusion_matrix = ConfusionMatrix(train= True)
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            P = torch.argmax(outputs, dim=1).tolist()
            T = labels.int().tolist()
            confusion_matrix.update(P,T)
            loss = criterion(outputs, labels.long())
            Loss += loss
            print(f"epoch:{epoch}batchsize:{i} loss is :", loss)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = confusion_matrix.plot(epoch)
        confusion_matrix.summary(epochs=epoch)
        model_save_path = os.path.join('./model_pth', f"{acc}_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        Mean_loss.append(Loss/(train_dataset.__len__()/32))
        '''
            This is val data set.
        '''
        print("#####################Start val data set.##########################")
        model.eval()
        confusion_matrix_ = ConfusionMatrix(train=False)
        with torch.no_grad():
            for i, (data_, labels_) in enumerate(val_dataloader):
                data_, labels_ = data_.to(device).float(), labels_.to(device)
                outputs_ = model(data_)
                P_ = torch.argmax(outputs_, dim=1).tolist()
                T_ = labels_.int().tolist()
                confusion_matrix_.update(P_,T_)
            confusion_matrix_.plot(epoch)
            confusion_matrix_.summary(epoch)
    Mean_Loss = []
    for a in Mean_loss:
        Mean_Loss.append(a.detach().cpu().numpy())
    Draw_loss_curve(epochs-1, Mean_Loss,run_time)
if __name__ == '__main__':
    train()