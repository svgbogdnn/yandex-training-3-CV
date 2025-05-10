'''
Из этого кода только 1 задача правильная
'''


import json
import os
import re
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths
DATA_DICT_PATH = "D:/Shit Delete/hw_overfitting_data_dict.npy"

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))

# Ensure hw_overfitting_data_dict is present
if not os.path.exists(DATA_DICT_PATH):
    import requests
    url = "https://github.com/girafe-ai/ml-course/raw/24f_ysda/homeworks/hw_overfitting/hw_overfitting_data_dict"
    print(f"Downloading data dict from {url} ...")
    r = requests.get(url)
    with open(DATA_DICT_PATH, 'wb') as f:
        f.write(r.content)

# Load the precomputed train/test arrays
loaded_data = np.load(DATA_DICT_PATH, allow_pickle=True).item()
train_np = torch.FloatTensor(loaded_data["train"])
test_np = torch.FloatTensor(loaded_data["test"])

# Load the FashionMNIST dataset
transform = torchvision.transforms.ToTensor()
train_dataset = FashionMNIST("D:/Shit Delete/", train=True, transform=transform, download=True)
test_dataset  = FashionMNIST("D:/Shit Delete/", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

def get_predictions(model, eval_data, step=10):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(eval_data), step):
            x = eval_data[i:i+step].to(device)
            y = model(x)
            preds.append(y.argmax(dim=1).cpu())
    preds = torch.cat(preds)
    return ",".join(str(int(x)) for x in preds)

def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# Task 1: Baseline model
model_task_1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
).to(device)

optimizer1 = torch.optim.Adam(model_task_1.parameters(), lr=1e-3)
criterion  = nn.CrossEntropyLoss()

for epoch in range(5):
    model_task_1.train()
    for x, y in tqdm(train_loader, desc=f"Task1 Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer1.zero_grad()
        logits = model_task_1(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer1.step()

print(f"Task1 Train acc: {get_accuracy(model_task_1, train_loader):.4f}, Test acc: {get_accuracy(model_task_1, test_loader):.4f}")

# Task 2: Model demonstrating overfitting (no Dropout/BatchNorm)
model_task_2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to(device)

optimizer2 = torch.optim.SGD(model_task_2.parameters(), lr=0.1)
for epoch in range(10):
    model_task_2.train()
    for x, y in tqdm(train_loader, desc=f"Task2 Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer2.zero_grad()
        loss = criterion(model_task_2(x), y)
        loss.backward()
        optimizer2.step()

print(f"Task2 Train acc: {get_accuracy(model_task_2, train_loader):.4f}, Test acc: {get_accuracy(model_task_2, test_loader):.4f}")

# Prepare submission for Tasks 1 & 2
submission_dict = {
    "train_predictions_task_1": get_predictions(model_task_1, train_np),
    "test_predictions_task_1" : get_predictions(model_task_1, test_np),
    "train_predictions_task_2": get_predictions(model_task_2, train_np),
    "test_predictions_task_2" : get_predictions(model_task_2, test_np),
}

with open("submission_dict_tasks_1_and_2.json", "w") as f:
    json.dump(submission_dict, f)
print("Saved submission_dict_tasks_1_and_2.json")

# Task 3: Introduce regularization to reduce overfitting
model_task_3 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
).to(device)

optimizer3 = torch.optim.Adam(model_task_3.parameters(), lr=1e-3)
for epoch in range(10):
    model_task_3.train()
    for x, y in tqdm(train_loader, desc=f"Task3 Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer3.zero_grad()
        loss = criterion(model_task_3(x), y)
        loss.backward()
        optimizer3.step()

print(f"Task3 Train acc: {get_accuracy(model_task_3, train_loader):.4f}, Test acc: {get_accuracy(model_task_3, test_loader):.4f}")

submission_dict.update({
    "train_predictions_task_3": get_predictions(model_task_3, train_np),
    "test_predictions_task_3" : get_predictions(model_task_3, test_np),
})

with open("submission_dict_final.json", "w") as f:
    json.dump(submission_dict, f)
print("Saved submission_dict_final.json")



'''
Из этого кода только 2 задача правильная
'''

import json
import os
import re

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

# --------------------------------
# Общие утилиты (не трогать)
# --------------------------------
def args_and_kwargs(*args, **kwargs):
    return args, kwargs

def parse_pytorch_model(model_str):
    def parse_layer(layer_str):
        name, params = layer_str.split("(", 1)
        info = {"type": name.strip()}
        template = layer_str.replace(name, "args_and_kwargs")
        args, kwargs = eval(template)
        if args or kwargs:
            info["parameters"] = {"args": args, **kwargs}
        else:
            info["parameters"] = {}
        return info

    lines = model_str.splitlines()
    model_name = lines[0].strip("()")
    layers = []
    rx = re.compile(r"\((\d+)\): (.+)")
    for line in lines[1:]:
        m = rx.match(line.strip())
        if m:
            idx, layer = m.groups()
            layers.append({"index": int(idx), "layer": parse_layer(layer)})
    return {"model_name": model_name, "layers": layers}

def get_predictions(model, data_tensor, step=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), step):
            batch = data_tensor[i : i + step].to(device)
            out = model(batch)
            preds.append(out.argmax(dim=1).cpu())
    return ",".join(map(str, torch.cat(preds).tolist()))

def get_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# --------------------------------
# Подготовка данных
# --------------------------------
DATA_FILE = "D:/Shit Delete/hw_overfitting_data_dict.npy"
assert os.path.exists(DATA_FILE), f"Скачайте и положите {DATA_FILE}"
data_dict = np.load(DATA_FILE, allow_pickle=True).item()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = torchvision.transforms.ToTensor()

train_ds = FashionMNIST(".", train=True,  transform=transform, download=True)
test_ds  = FashionMNIST(".", train=False, transform=transform, download=True)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

# ========================================
# Задача 1: простая CNN, test_acc ≥ 0.885
# ========================================
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1   = nn.Linear(32*7*7, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model_task_1 = Net1().to(device)
opt1   = optim.Adam(model_task_1.parameters(), lr=1e-3)
sched1 = StepLR(opt1, step_size=5, gamma=0.5)

for epoch in range(1, 16):
    model_task_1.train()
    pbar = tqdm(train_loader, desc=f"Task1 Epoch {epoch}/15", unit="batch")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model_task_1(x), y)
        opt1.zero_grad(); loss.backward(); opt1.step()
        pbar.set_postfix(loss=loss.item())
    sched1.step()

train_acc1 = get_accuracy(model_task_1, train_loader)
test_acc1  = get_accuracy(model_task_1, test_loader)

# Собираем словарь, **не сохраняем** отдельный файл для Task1
submission_dict = {
    "train_predictions_task_1": get_predictions(model_task_1, torch.FloatTensor(data_dict["train"])),
    "test_predictions_task_1":  get_predictions(model_task_1, torch.FloatTensor(data_dict["test"])),
    "model_task_1": parse_pytorch_model(str(model_task_1)),
}

# ========================================
# Задача 2: демонстрация переобучения
#           (no Dropout/BatchNorm)
# ========================================
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32,64, 3, padding=1)
        self.fc1   = nn.Linear(64*7*7,256)
        self.fc2   = nn.Linear(256,10)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model_task_2 = Net2().to(device)
opt2 = optim.SGD(model_task_2.parameters(), lr=0.01)

for epoch in range(1, 21):
    model_task_2.train()
    pbar = tqdm(train_loader, desc=f"Task2 Epoch {epoch}/20", unit="batch")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model_task_2(x), y)
        opt2.zero_grad(); loss.backward(); opt2.step()
        pbar.set_postfix(loss=loss.item())

train_acc2 = get_accuracy(model_task_2, train_loader)
test_acc2  = get_accuracy(model_task_2, test_loader)

submission_dict.update({
    "train_predictions_task_2": get_predictions(model_task_2, torch.FloatTensor(data_dict["train"])),
    "test_predictions_task_2":  get_predictions(model_task_2, torch.FloatTensor(data_dict["test"])),
    "model_task_2": parse_pytorch_model(str(model_task_2)),
})

# Сохраняем объединённый файл для задач 1+2
with open("submission_dict_tasks_1_and_2.json", "w") as fout:
    json.dump(submission_dict, fout, indent=2)
print("Saved → submission_dict_tasks_1_and_2.json")

# ========================================
# Задача 3: борьба с переобучением
#           (dropout/batchnorm, diff≤0.015)
# ========================================
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.fc1   = nn.Linear(64*7*7,256)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256,10)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x,2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

model_task_3 = Net3().to(device)
opt3   = optim.Adam(model_task_3.parameters(), lr=1e-3)
sched3 = StepLR(opt3, step_size=5, gamma=0.5)

for epoch in range(1, 16):
    model_task_3.train()
    pbar = tqdm(train_loader, desc=f"Task3 Epoch {epoch}/15", unit="batch")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model_task_3(x), y)
        opt3.zero_grad(); loss.backward(); opt3.step()
        pbar.set_postfix(loss=loss.item())
    sched3.step()

train_acc3 = get_accuracy(model_task_3, train_loader)
test_acc3  = get_accuracy(model_task_3, test_loader)


submission_dict.update({
    "train_predictions_task_3": get_predictions(model_task_3, torch.FloatTensor(data_dict["train"])),
    "test_predictions_task_3":  get_predictions(model_task_3, torch.FloatTensor(data_dict["test"])),
    "model_task_3": parse_pytorch_model(str(model_task_3)),
})

with open("submission_dict_final.json", "w") as fout:
    json.dump(submission_dict, fout, indent=2)
print("Saved → submission_dict_final.json")
