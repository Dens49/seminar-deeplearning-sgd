import os
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from gradvis import Visualization as vis, trajectory_plots as tplot, resnets
from gradvis.pytorch_nn_model import PyTorch_NNModel


def get_device():
    can_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if can_use_cuda else "cpu")
    cudnn.benchmark = True
    return device


class TrainVisCIFAR10:
    def __init__(self, cifar_data_dir):
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             self.normalizer])
        self.cifar_data_dir = cifar_data_dir
        self.trainset = torchvision.datasets.CIFAR10(root=self.cifar_data_dir, train=True,
                                                     download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=256, shuffle=True)
        self.device = get_device()

    def train_vis(self, model):
        """
        train_vis for cifar experiment
        """
        model.train()
        train_loss = 0
        criterion = nn.CrossEntropyLoss()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # break after 20 batches, because that's enough
            # to get a good estimate for visualization
            if batch_idx == 20:
                break
        return train_loss / (batch_idx + 1)

    def get_train_vis_function(self):
        return lambda model: self.train_vis(model)


def create_plots_from_checkpoints(bare_model, train_vis, checkpoints_path, checkpoint_base_filename, vis_output_dir, vis_output_base_filename, amount=-1, loss_3D_degrees=50):
    """
    Create static 2D and 3D loss visualizations for existing checkpoints
    """
    # check input and output dirs exist
    if not path_exists(checkpoints_path):
        print(
            f"Path '{checkpoints_path}' does not exist or isn't a directory. Did you even run a training?\nAborting...")
        return False

    if not path_exists(vis_output_dir):
        # TODO: this will probably not work in google colab...
        os.mkdir(vis_output_dir)
        print(f"directory {vis_output_dir} was created")

    # find torch device and checkpoints_paths to use
    device = get_device()
    checkpoints_filenames = get_filenames_in_dir(
        checkpoints_path, filename_filter=lambda x: checkpoint_base_filename in x)
    ordered_checkpoints_filenames = []
    for num in range(0, len(checkpoints_filenames)):
        ordered_checkpoints_filenames.append(
            checkpoint_base_filename + "_" + str(num))
    checkpoints_paths = list(map(lambda x: os.path.join(
        checkpoints_path, x), ordered_checkpoints_filenames))

    if amount == -1:
        checkpoint_path = checkpoints_paths[-1]
    elif amount < len(checkpoints_paths):
        # with for example 20 epochs there will be 21 checkpoints,
        # because 0 is before the first optimization step,
        # so in actuality amount + 1 steps will be rendered
        checkpoints_paths = checkpoints_paths[0:(amount + 1)]
        checkpoint_path = checkpoints_paths[amount]
    else:
        print(
            f"Amount {amount} too big. Only {len(checkpoints_paths) - 1} epoch checkpoints exist.\nAborting...")
        return False

    amount_used = len(checkpoints_paths)

    # restore model and create gradvis interface to the model
    model = restore_model(bare_model, checkpoint_path, device)
    nn_model = PyTorch_NNModel(model, train_vis, checkpoint_path)

    # create visualization
    vis_results_base_path = os.path.join(
        vis_output_dir, vis_output_base_filename)
    vis_results_path = vis_results_base_path + \
        "_" + str(amount_used - 1) + ".npz"
    vis.visualize(nn_model, checkpoints_paths, amount_used,
                  vis_results_path, proz=0.4, verbose=True)
    tplot.plot_loss_2D(vis_results_path,
                       filename=vis_results_base_path + "_2D_plot_pca_" + str(amount_used - 1))
    tplot.plot_loss_3D(vis_results_path,
                       filename=vis_results_base_path +
                       "_3D_plot_pca_" + str(amount_used - 1),
                       degrees=loss_3D_degrees)


def restore_model(bare_model, checkpoint_path, device):
    model = bare_model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model


def get_filenames_in_dir(path, filename_filter=lambda x: True):
    filenames = [filename for filename in os.listdir(
        path) if os.path.isfile(os.path.join(path, filename))]
    return list(filter(filename_filter, filenames))


def path_exists(path):
    return os.path.exists(path) and os.path.isdir(path)


if __name__ == "__main__":
    # TODO: implement interactive plot for notebooks
    bare_model = resnets.resnet20_cifar()
    train_vis_function = TrainVisCIFAR10(
        "./cifar_data").get_train_vis_function()

    create_plots_from_checkpoints(bare_model,
                                  train_vis_function,
                                  "./checkpoints/wip_cifar_resnet_sgd",
                                  "wip_cifar_resnet_sgd",
                                  "./visualizations/wip_cifar_resnet_sgd",
                                  "wip_cifar_resnet_sgd",
                                  amount=-1,
                                  loss_3D_degrees=100)
