import math
import time
from datetime import datetime
import os
import csv
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from gradvis import resnets


def main():
    ### START boilerplate section ###

    # find target device gpu or cpu
    can_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if can_use_cuda else "cpu")
    cudnn.benchmark = True
    print(f"running on device: {device}")

    # normalization as explained in https://pytorch.org/docs/stable/torchvision/models.html
    # and as in gradvis' resnet20_example.py
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [transforms.ToTensor(),
         normalizer])

    # define the train and test data
    cifar_data_dir = "./cifar_data"
    train_set = torchvision.datasets.CIFAR10(root=cifar_data_dir, train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=cifar_data_dir, train=False,
                                            download=True, transform=transform)

    # define default loaders for train and test data
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False)

    # define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # define model
    model = resnets.resnet20_cifar()
    model = model.to(device)

    # important: model should have been moved to GPU beofore defining the optimizer!
    # define optimizer
    default_sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    ### END boilerplate section ###

    def train_model(model):
        model.train()
        running_loss = 0
        correct_amount = 0

        batch_amount = math.ceil(
            len(train_loader.dataset) / train_loader.batch_size)
        pbar = tqdm(total=batch_amount, desc="  ")
        for batch_idx, (data, target) in enumerate(train_loader):
            # prepare
            inputs = data.to(device)
            labels = target.to(device)
            default_sgd_optimizer.zero_grad()

            # get loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # optimize
            loss.backward()
            default_sgd_optimizer.step()

            # get correct predictions
            _, predictions = torch.max(outputs, 1)
            correct_amount += torch.sum(predictions == labels.data)

            pbar.update(1)
        pbar.close()

        dataset_len = len(train_loader.dataset)
        avg_loss = running_loss / dataset_len
        accuracy = correct_amount.double() / dataset_len

        return avg_loss, accuracy.item()

    # TODO: very close to train_model --> refactor?
    def test_model(model):
        model.eval()
        running_loss = 0
        correct_amount = 0

        with torch.no_grad():
            for data, target in test_loader:
                # prepare
                inputs = data.to(device)
                labels = target.to(device)

                # get loss
                outputs = model(inputs)
                running_loss += criterion(outputs,
                                          labels).item() * inputs.size(0)

                # get correct predictions
                _, predictions = torch.max(outputs, 1)
                correct_amount += torch.sum(predictions == labels.data)

        dataset_len = len(test_loader.dataset)
        avg_loss = running_loss / dataset_len
        accuracy = correct_amount.double() / dataset_len

        return avg_loss, accuracy.item()

    def handle_statistics(**row):
        experiments_stats_path = "./experiments_stats"
        csv_path = os.path.join(
            experiments_stats_path, row["experiment_title"] + ".csv")
        file_exists = os.path.isfile(csv_path)

        if not (os.path.exists(experiments_stats_path) and os.path.isdir(experiments_stats_path)):
            # TODO: this will probably not work in google colab...
            os.mkdir(experiments_stats_path)
            print(f"directory {experiments_stats_path} was created")

        with open(csv_path, "a") as f:
            writer = csv.DictWriter(f, row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def get_latest_experiment_id(row):
        experiments_stats_path = "./experiments_stats"
        csv_path = os.path.join(
            experiments_stats_path, row["experiment_title"] + ".csv")

        latest_experiment_id = -1
        if os.path.isfile(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(
                    f, list(row.keys()).insert(0, "experiment_id"))
                # try reading one line only to see if the file has content
                for row in reader:
                    break
                # go back to start and discard first line because it's the header
                f.seek(0)
                f.readline()
                # find max experiment_id
                if reader.line_num > 0:
                    max_obj = max(reader, key=lambda x: x["experiment_id"])
                    latest_experiment_id = int(max_obj["experiment_id"])
        return latest_experiment_id

    def save_model_checkpoint(experiment_title, epoch):
        checkpoints_path = os.path.join("./checkpoints", experiment_title)

        if not (os.path.exists(checkpoints_path) and os.path.isdir(checkpoints_path)):
            # TODO: this will probably not work in google colab...
            os.mkdir(checkpoints_path)
            print(f"directory {checkpoints_path} was created")

        torch.save(model.state_dict(),
                   f"checkpoints/{experiment_title}/{experiment_title}_{epoch}")

    def train_epochs(experiment_title, epoch_amount=3):
        experiment_id = get_latest_experiment_id(
            dict(experiment_title=experiment_title)) + 1
        save_model_checkpoint(experiment_title, 0)

        start_secs = time.time()
        for epoch in range(1, epoch_amount + 1):
            # do the training
            print(f"Epoch: [{epoch}/{epoch_amount}]")
            train_avg_loss, train_acc = train_model(model)
            print("  Train", f"Avg Loss: {train_avg_loss:.4f}",
                  f"Acc: {train_acc:.4f}")
            print("  testing...")
            test_avg_loss, test_acc = test_model(model)
            print(
                "  Test", f"Avg Loss: {test_avg_loss:.4f}", f"Acc: {test_acc:.4f}")

            # save model checkpoint
            save_model_checkpoint(experiment_title, epoch)

            # save statistics
            epoch_stats = dict(
                experiment_id=experiment_id,
                experiment_title=experiment_title,
                epoch=epoch,
                epoch_amount=epoch_amount,
                train_avg_loss=train_avg_loss,
                train_acc=train_acc,
                test_avg_loss=test_avg_loss,
                test_acc=test_acc,
                timestamp=datetime.now()
            )
            handle_statistics(**epoch_stats)
        duration_secs = time.time() - start_secs
        print(f"Training took {duration_secs:.2f} seconds")

    # execute training
    train_epochs(experiment_title="wip_cifar_resnet_sgd", epoch_amount=20)


if __name__ == "__main__":
    main()
