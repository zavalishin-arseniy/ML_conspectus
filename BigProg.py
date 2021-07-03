# import torch
# # import numpy as np
# # from torch.autograd import Variable
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# #
# # #ПЛОХАЯ СЕТЬ
# # new_net = torch.nn.Sequential(
# #     #Задаём слой входной/внутренний/выходной
# #     torch.nn.Linear(5, 2),
# #     #Задаём функцию активации в этом слое
# #     torch.nn.ReLU(),
# #     #Задаём следующий слой и так столько сколько нужно
# #     torch.nn.Linear(2, 10),
# #     torch.nn.Softmax(),
# # )
# #
# # # выбираем часть датасета для обуения
# # def batch_gen(X, y, batch_size=1):
# #     idx=np.random.randint(X.shape, size=batch_size)
# #     x_batch=X[idx]
# #     y_batch=y[idx]
# #     return Variable(torch.FloatTensor(x_batch)), Variable(torch.LongTensor(y_batch))
# #
# # #Задаём функцию потерь и оптимизатор
# #
# # loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
# #
# # learning_rate = 1e-4
# # optimizer = torch.optim.SGD(new_net.parameters(), lr=learning_rate)
# #
# # #обучение
# #
# # for i in range(10000):
# #     x_b, y_b = batch_gen(X, y)
# #
# #     #forward прогоняем данные через сеть
# #     y_pred = new_net(x_b)
# #
# #     #loss считаем функцию потерь
# #     loss = loss_fn(y_pred, y_b)
# #
# #     #Зануляем!
# #     optimizer.zero_grad()
# #
# #     #backward обратное распространение ошибки
# #     loss.backward()
# #
# #     #Обновляем!
# #     optimizer.step()
# #
# # #СВЕРТОЧНАЯ СЕТЬ

import torch
import torchvision
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as plt


#matplotlib inline

num_epochs = 25
num_classes = 10
batch_size = 250
learning_rate = 0.001
momentum = 0

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

DATA_PATH = 'C:UsersUserDocumentsML'
MODEL_STORE_PATH = 'C:UsersUserDocumentsML'


train_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, transform=trans)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.drop_out = nn.Dropout()
#         self.fc1 = nn.Linear(2 * 2 * 64, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.drop_out(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         out = self.fc3(out)
#         out = self.fc4(out)
#         return out
#
#
# model = ConvNet()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 30, 5)
        self.fc1 = nn.Linear(5 * 5 * 30, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,30*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return x


model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Прямой запуск
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Обратное распространение и оптимизатор
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Отслеживание точности
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Сохраняем модель и строим график
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')