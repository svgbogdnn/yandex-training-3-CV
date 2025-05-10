import json
import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST


# Функции для получения предсказаний и вычисления точности
def get_predictions(model, eval_data, step=10):
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(eval_data), step):
            y_predicted = model(eval_data[idx: idx + step].to(device))
            predicted_labels.append(y_predicted.argmax(dim=1).cpu())
    predicted_labels = torch.cat(predicted_labels)
    predicted_labels = ",".join([str(x.item()) for x in list(predicted_labels)])
    return predicted_labels


def get_accuracy(model, data_loader):
    predicted_labels = []
    real_labels = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            y_predicted = model(batch[0].to(device))
            predicted_labels.append(y_predicted.argmax(dim=1).cpu())
            real_labels.append(batch[1])
    predicted_labels = torch.cat(predicted_labels)
    real_labels = torch.cat(real_labels)
    accuracy_score = (predicted_labels == real_labels).type(torch.FloatTensor).mean()
    return accuracy_score


# Проверка наличия файла с данными
assert os.path.exists("D:/Shit Delete/hw_overfitting_data_dict.npy"), \
    "Please, download `hw_overfitting_data_dict.npy` and place it in the working directory"

CUDA_DEVICE_ID = 0
device = torch.device(f"cuda:{CUDA_DEVICE_ID}") if torch.cuda.is_available() else torch.device("cpu")


# Класс модели
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model_task_1 = SimpleCNN().to(device)

if __name__ == '__main__':
    # Для Windows рекомендуется оборачивать основное тело кода в блок main.
    # Если возникают проблемы с multiprocessing, задайте num_workers=0
    train_fmnist_data = FashionMNIST(
        ".", train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    test_fmnist_data = FashionMNIST(
        ".", train=False, transform=torchvision.transforms.ToTensor(), download=True
    )

    # При проблемах с DataLoader можно попробовать num_workers=0
    train_data_loader = torch.utils.data.DataLoader(
        train_fmnist_data, batch_size=32, shuffle=True, num_workers=2
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_fmnist_data, batch_size=32, shuffle=False, num_workers=2
    )

    # Для визуализации примера изображения
    random_batch = next(iter(train_data_loader))
    _image, _label = random_batch[0][0], random_batch[1][0]
    plt.figure()
    plt.imshow(_image.reshape(28, 28))
    plt.title(f"Image label: {_label}")
    plt.show()

    # Проверка работы модели
    try:
        x = random_batch[0].to(device)
        y_predicted = model_task_1(x)
    except Exception as e:
        print("Something is wrong with the model")
        raise e

    assert y_predicted.shape[-1] == 10, "Model should predict 10 logits/probas"
    print("Everything seems fine!")

    # Обучение модели
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_task_1.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model_task_1.train()
        running_loss = 0.0
        for inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_task_1(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_fmnist_data)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Оценка качества модели
    train_acc_task_1 = get_accuracy(model_task_1, train_data_loader)
    test_acc_task_1 = get_accuracy(model_task_1, test_data_loader)
    print(f"Train accuracy: {train_acc_task_1:3.5}")
    print(f"Test accuracy: {test_acc_task_1:3.5}")

    # Проверка пороговых значений точности
    assert test_acc_task_1 >= 0.885, "Test accuracy is below the 0.885 threshold"
    assert train_acc_task_1 >= 0.905, "Train accuracy is below the 0.905 threshold"

    # Генерация файла с предсказаниями
    loaded_data_dict = np.load("D:/Shit Delete/hw_fmnist_data_dict.npy", allow_pickle=True)
    submission_dict = {
        "train_predictions_task_1": get_predictions(
            model_task_1, torch.FloatTensor(loaded_data_dict.item()["train"])
        ),
        "test_predictions_task_1": get_predictions(
            model_task_1, torch.FloatTensor(loaded_data_dict.item()["test"])
        ),
    }

    with open("submission_dict_fmnist_task_1_(2).json", "w") as f:
        json.dump(submission_dict, f)
    print("File saved to `submission_dict_fmnist_task_1_(2).json`")

'''
100%|██████████| 26.4M/26.4M [01:27<00:00, 301kB/s] 
100%|██████████| 29.5k/29.5k [00:00<00:00, 604kB/s]
100%|██████████| 4.42M/4.42M [00:01<00:00, 4.38MB/s]
100%|██████████| 5.15k/5.15k [00:00<?, ?B/s]
Everything seems fine!
Epoch 1/10, Loss: 0.4686
Epoch 2/10, Loss: 0.3134
Epoch 3/10, Loss: 0.2772
Epoch 4/10, Loss: 0.2526
Epoch 5/10, Loss: 0.2343
Epoch 6/10, Loss: 0.2207
Epoch 7/10, Loss: 0.2074
Epoch 8/10, Loss: 0.1956
Epoch 9/10, Loss: 0.1886
Epoch 10/10, Loss: 0.1807
Train accuracy: 0.94022
Test accuracy: 0.9121
File saved to `submission_dict_fmnist_task_1_(2).json`
'''