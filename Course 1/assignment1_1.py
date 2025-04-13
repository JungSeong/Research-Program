import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) # GPU를 사용할 수 있는지 확인

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,], std=[0.5,])
])

train_dataset = torchvision.datasets.FashionMNIST("FashionMNIST/", download=True, train=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST("FashionMNIST/", download=True, train=False, transform=transform)

target_labels = [1, 4, 7, 8]

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

original_labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}

my_labels_map = {'Trouser': 0, 'Coat': 1, 'Sneaker': 2, 'Bag': 3}

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
    plt.imshow(img[0, :, :], cmap='grey')
    plt.title(f'{label} : {class_name}')
plt.show()

class FashionDNN(nn.Module):
    def __init__(self):
        super(FashionDNN,self).__init__()
        self.fc1 = nn.Linear(in_features=784,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=64)
        self.fc3 = nn.Linear(in_features=64,out_features=32)
        self.fc4 = nn.Linear(in_features=32,out_features=4)

    def forward(self,input_data):
        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

learning_rate = 1e-4;
model = FashionDNN();
model.to(device)

criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate);

print(model)
print(optimizer)

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
    model.train()  # 모델을 학습 모드로 설정
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        train = images.view(-1, 1, 28, 28)
        
        outputs = model(train)
        loss = criterion(outputs, labels)

        train_predictions = torch.max(outputs, 1)[1]
        train_correct += (train_predictions == labels).sum().item()
        train_total += labels.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()  # 미니배치 손실 합산

    # Validation Phase
    model.eval()  # 모델을 평가 모드로 설정 (Dropout 등 비활성화)
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():  # 기울기 업데이트 비활성화 (메모리 절약)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test = images.view(-1, 1, 28, 28)
            outputs = model(test)
            
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

# 훈련된 모델의 테스트를 위한 함수의 작성

def test_model(image) :
    model.eval()
    with torch.no_grad() :
        image = image.to(device)
        test = image.view(-1, 1, 28, 28)
        output = model(test)
        predicted_label = torch.argmax(output, 1).item()
        predicted_name = inv_my_labels_map[predicted_label]

        image = image.to('cpu')
        plt.imshow(image[0, :, :], cmap='grey')
        plt.title(f'{predicted_label} : {predicted_name}')

img0, label0 = test_subset_dataset[0]
test_model(img0)    

img1, label1 = test_subset_dataset[3]
test_model(img1)  

img2, label2 = test_subset_dataset[4]
test_model(img2)   

img3, label3 = test_subset_dataset[10]
test_model(img3)