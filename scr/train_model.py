import mlflow.pytorch
import mlflow
from torchvision.datasets import MNIST
from models import Net
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
mlflow.set_tracking_uri('http://127.0.0.1:5000')


def train(model, train_loader, optimizer):
    model.train()
    correct = 0  # Счетчик правильных предсказаний
    total = 0    # Общее количество примеров
    for i, (data, target) in enumerate(train_loader):
        print(i, len(train_loader))
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # Подсчет метрик
        _, predicted = output.max(1)  # Получаем предсказанные классы
        total += target.size(0)       # Обновляем общее количество примеров
        correct += predicted.eq(target).sum().item()  # Обновляем счетчик правильных предсказаний

    accuracy = correct / total  # Вычисляем точность
    return accuracy  # Возвращаем точность


def train_and_log_model(lr):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Загружаем набор данных MNIST
    dataset1 = MNIST("../data", train=True, download=True, transform=transform)

    # Создаем DataLoader с ограниченным набором данных
    train_loader = DataLoader(dataset1, batch_size=64, shuffle=True)

    print('Dataset собран')
    with mlflow.start_run():
        mlflow.log_param('lr', lr)
        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        accuracy = train(model, train_loader, optimizer)  # Получаем точность
        mlflow.log_metric('accuracy', accuracy)  # Логируем точность
        run_id = mlflow.active_run()._info.run_id
        model_uri = f"runs:/{run_id}/mnist_app"
        registered_model = mlflow.register_model(model_uri, "Mnist_app")
        client = mlflow.MlflowClient()

        client.set_model_version_tag(
            name="Mnist_app",
            version=registered_model.version,
            key='accuracy',
            value=str(round(accuracy, 3))
        )


def promote_best_model(model_name):
    client = mlflow.MlflowClient()
    best_accuracy = 0
    best_version = None
    for version in client.search_registered_models(f"name='{model_name}'"):
        tmp_accuracy = version.tags.get('accuracy')
        if tmp_accuracy:
            tmp_accuracy = float(tmp_accuracy)
            if tmp_accuracy > best_accuracy:
                best_accuracy = tmp_accuracy
                best_version = version
    if best_version:
        client.transition_model_version_stage(
            name=best_version.name,
            version=best_version,
            stage='Production'
        )


if __name__ == '__main__':
    mlflow.end_run()
    train_and_log_model(0.1)
    train_and_log_model(0.01)
    train_and_log_model(0.001)
    promote_best_model('Mnist_app')