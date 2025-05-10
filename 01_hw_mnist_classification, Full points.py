import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
from IPython.display import clear_output
import os
import json

# Создаем модель глобально — это не запускает подпроцессы
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

if __name__ == '__main__':
    # Устанавливаем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Загружаем данные. Если проблема с multiprocessing сохраняется, попробуйте num_workers=0
    train_mnist_data = MNIST('.', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_mnist_data = MNIST('.', train=False, transform=torchvision.transforms.ToTensor(), download=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_mnist_data,
        batch_size=32,
        shuffle=True,
        num_workers=2  # можно заменить на 0, если возникают ошибки
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_mnist_data,
        batch_size=32,
        shuffle=False,
        num_workers=2  # можно заменить на 0, если возникают ошибки
    )

    # Получаем случайную партию для визуализации и теста
    random_batch = next(iter(train_data_loader))
    _image, _label = random_batch[0][0], random_batch[1][0]
    plt.figure()
    plt.imshow(_image.reshape(28, 28))
    plt.title(f'Image label: {_label}')
    plt.show()

    # Тест: проверяем работу модели на случайной партии
    try:
        x = random_batch[0].reshape(-1, 784).to(device)
        y_predicted = model(x)
    except Exception as e:
        print('Something is wrong with the model')
        raise e

    # Теперь y_predicted определена – проверка пройдет успешно
    assert y_predicted.shape[-1] == 10, 'Model should predict 10 logits/probas'
    print('Everything seems fine with the model!')

    # Обучение модели
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_data_loader:
            inputs, labels = batch
            inputs = inputs.reshape(-1, 784).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_mnist_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Оценка точности на обучающем наборе
    predicted_labels = []
    real_labels = []
    model.eval()
    with torch.no_grad():
        for batch in train_data_loader:
            inputs, labels = batch
            outputs = model(inputs.reshape(-1, 784).to(device))
            predicted_labels.append(outputs.argmax(dim=1).cpu())
            real_labels.append(labels)

    predicted_labels = torch.cat(predicted_labels)
    real_labels = torch.cat(real_labels)
    train_acc = (predicted_labels == real_labels).type(torch.FloatTensor).mean()
    print(f'Neural network accuracy on train set: {train_acc:3.5}')

    # Оценка точности на тестовом наборе
    predicted_labels = []
    real_labels = []
    with torch.no_grad():
        for batch in test_data_loader:
            inputs, labels = batch
            outputs = model(inputs.reshape(-1, 784).to(device))
            predicted_labels.append(outputs.argmax(dim=1).cpu())
            real_labels.append(labels)

    predicted_labels = torch.cat(predicted_labels)
    real_labels = torch.cat(real_labels)
    test_acc = (predicted_labels == real_labels).type(torch.FloatTensor).mean()
    print(f'Neural network accuracy on test set: {test_acc:3.5}')

    # Проверка пороговых значений
    assert test_acc >= 0.92, 'Test accuracy is below 0.92 threshold'
    assert train_acc >= 0.91, 'Train accuracy is below 0.91 while test accuracy is fine. Please, check your model and data flow'

    # Функция для получения предсказаний
    def get_predictions(model, eval_data, step=10):
        predicted_labels = []
        model.eval()
        with torch.no_grad():
            for idx in range(0, len(eval_data), step):
                outputs = model(eval_data[idx:idx + step].reshape(-1, 784).to(device))
                predicted_labels.append(outputs.argmax(dim=1))
        predicted_labels = torch.cat(predicted_labels).numpy()
        predicted_labels = ','.join([str(x) for x in list(predicted_labels)])
        return predicted_labels

    # Проверка наличия файла с данными
    data_file = 'D:/Shit Delete/hw_mnist_data_dict.npy'
    assert os.path.exists(data_file), f'Please, download `hw_mnist_data_dict.npy` and place it in the working directory: {data_file}'

    loaded_data_dict = np.load(data_file, allow_pickle=True)

    submission_dict = {
        'train': get_predictions(model, torch.FloatTensor(loaded_data_dict.item()['train'])),
        'test': get_predictions(model, torch.FloatTensor(loaded_data_dict.item()['test']))
    }

    submission_file = 'submission_dict_mnist_task_1_(1)     .json'
    with open(submission_file, 'w') as iofile:
        json.dump(submission_dict, iofile)
    print(f'File saved to `{submission_file}`')


'''
Everything seems fine with the model!
Epoch 1/10, Loss: 0.2445
Epoch 2/10, Loss: 0.0965
Epoch 3/10, Loss: 0.0656
Epoch 4/10, Loss: 0.0486
Epoch 5/10, Loss: 0.0375
Epoch 6/10, Loss: 0.0298
Epoch 7/10, Loss: 0.0254
Epoch 8/10, Loss: 0.0220
Epoch 9/10, Loss: 0.0191
Epoch 10/10, Loss: 0.0161
Neural network accuracy on train set: 0.99528
Neural network accuracy on test set: 0.9791
File saved to `D:/Shit Delete/submission_dict_mnist_task_1.json`
'''