import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,], std=[0.5,])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) # GPU를 사용할 수 있는지 확인

train_dataset = torchvision.datasets.CIFAR10("CIFAR10/", download=True, train=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10("CIFAR10/", download=True, train=False, transform=transform)

target_labels = [0, 1, 8]

train_indices = [idx for idx, (img, label) in enumerate(train_dataset) if label in target_labels]
test_indices = [idx for idx, (img, label) in enumerate(test_dataset) if label in target_labels]

from torch.utils.data import Subset

train_subset_dataset = Subset(train_dataset, train_indices)
test_subset_dataset = Subset(test_dataset, test_indices)

class RemappedLabelDataset(Dataset):
    def __init__(self, subset_dataset, original_label_map, new_label_map):
        self.subset_dataset = subset_dataset
        self.original_label_map = original_label_map
        self.new_label_map = new_label_map

    def __len__(self):
        return len(self.subset_dataset)

    def __getitem__(self, idx):
        image, label = self.subset_dataset[idx]
        class_name = self.original_label_map[label]
        new_label = self.new_label_map[class_name]
        return image, new_label

original_labels_map = {0 : 'airplane', 1 : 'automobile', 2 : 'bird', 3 : 'cat', 4 : 'deer', 5 : 'dog', 6 : 'frog',
              7 : 'horse', 8 : 'ship', 9 : 'truck'}

my_labels_map = {'airplane': 0, 'automobile': 1, 'ship': 2}

train_subset_dataset = RemappedLabelDataset(train_subset_dataset, original_labels_map, my_labels_map)
test_subset_dataset = RemappedLabelDataset(test_subset_dataset, original_labels_map, my_labels_map)

train_loader = torch.utils.data.DataLoader(train_subset_dataset, 
                                           batch_size=200, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_subset_dataset,
                                          batch_size=200)

fig = plt.figure(figsize=(8,8));
columns = 4;
rows = 5;

inv_my_labels_map = {v: k for k, v in my_labels_map.items()}

# 0 : Trouser, 1 : Coat, 2 : Sneaker, 3 : Bag로 잘 구성된 것을 확인할 수 있다
for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(train_subset_dataset));
    img, label = train_subset_dataset[img_xy]
    class_name = inv_my_labels_map[label]

    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img[0, :, :], cmap='gray')
    plt.title(f'{label} : {class_name}')
plt.show()

# 1. 새로 추가된 fc1 레이어에 대해서만 추가 학습을 진행

from torchvision import models

vgg16 = models.vgg16(pretrained=True)

class MyVGG16Net(nn.Module):
    def __init__(self):
        super(MyVGG16Net, self).__init__()
        # 기존의 VGG16 모델을 base model로써 사용
        base_model = models.vgg16(pretrained=True)
        self.features = base_model.features

        # 새로운 classifier의 정의
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096), # MaxPooling을 5번 거쳤으므로 이미지의 가로, 세로 사이즈가 2^5만큼 작아지게 된다
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

my_model = MyVGG16Net()
my_model.to(device)

for param in my_model.parameters():
    param.requires_grad = False # frozen
    
for param in my_model.fc.parameters():
    param.requires_grad = True # 마지막 레이어는 autograd 활성화

print(my_model)

learning_rate = 4 * 1e-06

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

num_epochs = 20
count = 0
train_loss_list = []
test_loss_list = []

train_accuracy_list = []
test_accuracy_list = []

for epoch in range(num_epochs):
    train_loss_epoch = 0
    test_loss_epoch = 0
    
    # Training Phase
    my_model.train()  # 모델을 학습 모드로 설정
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        train = images.view(-1, 3, 32, 32)
        
        outputs = my_model(train)
        loss = criterion(outputs, labels)

        train_predictions = torch.max(outputs, 1)[1]
        train_correct += (train_predictions == labels).sum().item()
        train_total += labels.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()  # 미니배치 손실 합산

    # Validation Phase
    my_model.eval()  # 모델을 평가 모드로 설정 (Dropout 등 비활성화)
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():  # 기울기 업데이트 비활성화 (메모리 절약)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test = images.view(-1, 3, 32, 32)
            outputs = my_model(test)
            
            loss = criterion(outputs, labels)
            test_loss_epoch += loss.item()  # 미니배치 손실 합산
            
            test_predictions = torch.max(outputs, 1)[1]
            test_correct += (test_predictions == labels).sum().item()
            test_total += labels.size(0)

    train_accuracy = train_correct * 100 / train_total
    test_accuracy = test_correct * 100 / test_total

    train_loss_list.append(train_loss_epoch / len(train_loader))  # 평균 손실 저장
    test_loss_list.append(test_loss_epoch / len(test_loader))  # 평균 손실 저장

    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)

    print(f"Epoch {epoch+1}, Train_Accuracy: {train_accuracy}%, Test_Accuracy: {test_accuracy}%")

# plot loss graph

plt.plot(range(num_epochs), train_loss_list, label="Train Loss")
plt.plot(range(num_epochs), test_loss_list, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.show()

# plot accuracy graph

plt.plot(range(num_epochs), train_accuracy_list, label="Train Accuracy")
plt.plot(range(num_epochs), test_accuracy_list, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.show()

# 2. 전체 Layer에 대해서 학습을 진행

from torchvision import models

vgg16 = models.vgg16(pretrained=True)

class MyVGG16Net_2(nn.Module):
    def __init__(self):
        super(MyVGG16Net_2, self).__init__()
        # 기존의 VGG16 모델을 base model로써 사용
        base_model = models.vgg16(pretrained=True)
        self.features = base_model.features

        # 새로운 classifier의 정의
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096), # MaxPooling을 5번 거쳤으므로 이미지의 가로, 세로 사이즈가 2^5만큼 작아지게 된다
            nn.ReLU(),
            nn.Linear(4096, 500),
            nn.ReLU(),
            nn.Linear(500, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

my_model_2 = MyVGG16Net_2()
my_model_2.to(device)

for param in my_model_2.parameters():
    param.requires_grad = True 
    
for param in my_model_2.fc.parameters():
    param.requires_grad = True 

print(my_model_2)

learning_rate = 4 * 1e-06

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model_2.parameters(), lr=learning_rate)

num_epochs = 20
count = 0
train_loss_list = []
test_loss_list = []

train_accuracy_list = []
test_accuracy_list = []

for epoch in range(num_epochs):
    train_loss_epoch = 0
    test_loss_epoch = 0
    
    # Training Phase
    my_model_2.train()  # 모델을 학습 모드로 설정
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        train = images.view(-1, 3, 32, 32)
        
        outputs = my_model_2(train)
        loss = criterion(outputs, labels)

        train_predictions = torch.max(outputs, 1)[1]
        train_correct += (train_predictions == labels).sum().item()
        train_total += labels.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()  # 미니배치 손실 합산

    # Validation Phase
    my_model_2.eval()  # 모델을 평가 모드로 설정 (Dropout 등 비활성화)
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():  # 기울기 업데이트 비활성화 (메모리 절약)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test = images.view(-1, 3, 32, 32)
            outputs = my_model_2(test)
            
            loss = criterion(outputs, labels)
            test_loss_epoch += loss.item()  # 미니배치 손실 합산
            
            test_predictions = torch.max(outputs, 1)[1]
            test_correct += (test_predictions == labels).sum().item()
            test_total += labels.size(0)

    train_accuracy = train_correct * 100 / train_total
    test_accuracy = test_correct * 100 / test_total

    train_loss_list.append(train_loss_epoch / len(train_loader))  # 평균 손실 저장
    test_loss_list.append(test_loss_epoch / len(test_loader))  # 평균 손실 저장

    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)

    print(f"Epoch {epoch+1}, Train_Accuracy: {train_accuracy}%, Test_Accuracy: {test_accuracy}%")

# plot loss graph

plt.plot(range(num_epochs), train_loss_list, label="Train Loss")
plt.plot(range(num_epochs), test_loss_list, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.show()

# plot accuracy graph

plt.plot(range(num_epochs), train_accuracy_list, label="Train Accuracy")
plt.plot(range(num_epochs), test_accuracy_list, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.show()