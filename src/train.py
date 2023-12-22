
# import the necessary libraries
import warnings
import os  # nopep8
os.sys.path.append("")  # nopep8
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models
from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import numpy as np
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import torchvision
from torch.autograd import Variable
from PIL import Image
import PIL.ImageOps
import os
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from src.logger import create_logger

torch.cuda.is_available()


run_id = 4
train_dir = "data/train"
test_dir = "data/test"

train_csv = "data/spam_train.csv"
test_csv = "data/spam_test.csv"

models_path = f"models/{run_id}"
logfile = f"logs/{run_id}.log"
logger = create_logger(logfile)

os.makedirs(models_path, exist_ok=True)

epochs = 100
max_lr = 0.001
batch_size = 64
num_worker = 16
save_model_every = 10
early_stopping = 10

resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features

resnet.fc1 = nn.Sequential(
    nn.Linear(num_ftrs, 256)
)
# resnet.fc2 = nn.Sequential(
#     nn.Linear(24, 2)
# )

resnet.classifier = nn.Sequential(
    resnet.fc1,
    # nn.Dropout(p=0.2),
    # resnet.fc2
)
# resnet.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 256),
#     nn.Dropout(p=0.1),
#     nn.Linear(256, 2)
# )

transforms = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    # transforms.RandomCrop((110, 110)),
    # transforms.ColorJitter(brightness=0.5, contrast=0.1,
    #                        saturation=0.1, hue=0.1),
    transforms.ToTensor()
])
# preprocessing and loading the dataset


class SiameseData(Dataset):
    def __init__(self, train_csv=None, train_dir=None, transform=None):
        self.train_df = pd.read_csv(train_csv)
        self.train_df.columns = ['image1', 'image2', 'image3']
        self.train_dir = train_dir
        self.transform = transform

    def __getitem__(self, index):
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        img1 = img1.convert("L").convert("RGB")
        img2 = img2.convert("L").convert("RGB")
        # apply image transormations
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([int(self.train_df.iat[index, 2])], dtype=np.float32))

    def __len__(self):
        return len(self.train_df)


class ContrastiveLoss(nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=4.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(
            output1, output2, keepdim=True)
        
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = resnet

    def forward(self, input1, input2):
        # forward pass of input 1
        # output1 = self.forward_once(input1)
        output1 = self.cnn(input1)
        # forward pass of input 2
        # output2 = self.forward_once(input2)
        output2 = self.cnn(input2)
        output1 = output1.view(output1.size()[0], -1)
        output2 = output2.view(output2.size()[0], -1)
        return output1, output2


avr_loss_train = []
avr_loss_test = []
accuracy_test = []
accuracy_train = []
# iteration_number = 0


def calculate_accuracy(output1, output2, label):
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_test_contrastive = criterion(output1, output2, label)
    predictions = (euclidean_distance < 2).float()
    correct_predictions = (predictions != label).float()
    accuracy = torch.mean(correct_predictions).item() * 100
    return accuracy
# train the model


def train(epochs, max_lr, model, train_dl, test_dl, opt_func=torch.optim.Adam):
    global loss, avr_loss
    torch.cuda.empty_cache()
    optimizer = opt_func(resnet.parameters(), max_lr, weight_decay=0.0001)
    train_accuracies = []
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_dl))
    for epoch in range(1, epochs+1):
        logger.info(f"Training epoch: {epoch}")
        losses_train = []
        losses_test = []
        accuracys = []
        accuracys1 = []

        for i, data in enumerate(train_dl, 0):

            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            euclidean_distance = F.pairwise_distance(
                output1, output2, keepdim=True)
            loss_train_contrastive = criterion(output1, output2, label)
            loss_train_contrastive.backward()

            optimizer.step()
            accuracys1.append(calculate_accuracy(output1, output2, label))

            losses_train.append(loss_train_contrastive.item())

        losses_train = np.array(losses_train)
        avr_loss_train.append(math.log(losses_train.mean()/len(train_dl)))

        arg_accuracy1 = sum(accuracys1) / len(accuracys1)
        accuracy_train.append(arg_accuracy1)

        logger.info(
            f"\tAvarage loss train: {losses_train.mean()/len(train_dl):.4f}\t Accuracy: {arg_accuracy1:.2f}")
        
        if epoch % save_model_every == 0:
            filename = os.path.join(models_path, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), filename)


        for i, data in enumerate(test_dl, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = model(img0, img1)
            euclidean_distance = F.pairwise_distance(
                output1, output2, keepdim=True)
            loss_test_contrastive = criterion(output1, output2, label)
            losses_test.append(loss_test_contrastive.item())

            predictions = (euclidean_distance < 2).float()
            correct_predictions = (predictions != label).float()
            accuracy = torch.mean(correct_predictions).item() * 100
            accuracys.append(accuracy)



        losses_test = np.array(losses_test)
        avr_loss_test.append(math.log(losses_test.mean()/len(test_dl)))

        arg_accuracy = sum(accuracys) / len(accuracys)
        accuracy_test.append(arg_accuracy)



        logger.info(
            f"\tAvarage loss test: {losses_test.mean()/len(test_dl):.4f}\t Accuracy: {arg_accuracy:.2f}")
        if arg_accuracy == max(accuracy_test):
            print("Saved new best model\n")
            filename = os.path.join(models_path, f"best_model.pt")
            torch.save(model.state_dict(), filename)
    return model


warnings.simplefilter("ignore")


opt_func = torch.optim.Adam
# set device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = SiameseNetwork()
model = model.to(device)

criterion = ContrastiveLoss()
# criterion = nn.CrossEntropyLoss()

train_ds = SiameseData(train_csv, train_dir, transform=transforms)
train_dl = DataLoader(train_ds, shuffle=True, num_workers=num_worker,
                      pin_memory=True, batch_size=batch_size)

test_ds = SiameseData(test_csv, test_dir, transform=transforms)
test_dl = DataLoader(test_ds, shuffle=True, num_workers=num_worker,
                     pin_memory=True, batch_size=batch_size)
os.environ['WANDB_CONSOLE'] = 'off'
history = train(epochs, max_lr, model, train_dl, test_dl, opt_func)
