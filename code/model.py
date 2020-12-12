from pathlib import Path

import PIL
import pytorch_lightning as pl
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as transforms


class Stn(nn.Module):
    def __init__(self, setup_dict):
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                setup_dict["conv1_in"],
                setup_dict["conv1_out"],
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                setup_dict["conv2_in"],
                setup_dict["conv2_out"],
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(setup_dict["lin1_in"], setup_dict["lin1_out"]),
            nn.ReLU(),
            nn.Linear(setup_dict["lin2_in"], setup_dict["lin2_out"]),
        )

    def forward(self, x):
        xs = self.loc_net(x)
        theta = xs.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class LitModel(pl.LightningModule):
    def __init__(self, hparams, mean, std, train_dataset, test_dataset):
        super(LitModel, self).__init__()
        self.hparams = hparams
        self.mean = mean
        self.std = std
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.activation_function_features = (
            nn.ELU()
            if self.hparams.activation_function_features == "ELU"
            else nn.ReLU()
        )
        self.activation_function_features = (
            nn.ELU()
            if self.hparams.activation_function_classifier == "ELU"
            else nn.ReLU()
        )

        self.features = nn.Sequential(
            # Hidden 1
            Stn(self.hparams.stn_parameter[0]),
            nn.Conv2d(3, 200, (7, 7), stride=1, padding=2),
            self.activation_function_features,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(self.hparams.dropout_rate),
            nn.BatchNorm2d(200),
            # Hidden 2
            Stn(self.hparams.stn_parameter[1]),
            nn.Conv2d(200, 250, (4, 4), stride=1, padding=2),
            self.activation_function_features,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(self.hparams.dropout_rate),
            nn.BatchNorm2d(250),
            # Hidden 3
            Stn(self.hparams.stn_parameter[2]),
            nn.Conv2d(250, 350, (4, 4,), stride=1, padding=2),
            self.activation_function_features,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(self.hparams.dropout_rate),
            nn.BatchNorm2d(350),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12600, 400),
            self.activation_function_features,
            nn.Linear(400, 43),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

    def configure_optimizers(self):
        if self.hparams.optimizer == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                nesterov=True
            )
        print("\n")
        print(optimizer)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), y.cpu())
        val_acc = torch.tensor(val_acc)

        return {"val_loss": loss, "val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        print("\n")
        print("Val loss: ", avg_loss)
        print("val acc: ", avg_val_acc)
        print("\n")
        tensorboard_logs = {"val_loss": avg_loss, "avg_val_acc": avg_val_acc}
        return {"val_loss": avg_loss, "progress_bar": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), y.cpu())

        return {"test_acc": torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        tensorboard_logs = {"avg_test_acc": avg_test_acc}
        return {
            "avg_test_acc": avg_test_acc,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def train_dataloader(self):
        train_set_normal = torchvision.datasets.ImageFolder(
            root=str(Path(self.train_dataset).joinpath("train")),
            transform=transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.RandomAffine(
                                0, translate=(0.2, 0.2), resample=PIL.Image.BICUBIC
                            ),
                            transforms.RandomAffine(
                                0, shear=20, resample=PIL.Image.BICUBIC
                            ),
                            transforms.RandomAffine(
                                0, scale=(0.8, 1.2), resample=PIL.Image.BICUBIC
                            ),
                        ]
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
        )

        loader = DataLoader(
            train_set_normal, batch_size=50, num_workers=8, shuffle=True
        )

        return loader

    def val_dataloader(self):
        val_set_normal = torchvision.datasets.ImageFolder(
            root=str(Path(self.train_dataset).joinpath("val")),
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(self.mean, self.std),]
            ),
        )

        val_loader = DataLoader(val_set_normal, batch_size=50, num_workers=8)

        return val_loader

    def test_dataloader(self):
        test_set_normal = torchvision.datasets.ImageFolder(
            root=str(self.test_dataset),
            transform=transforms.Compose(
                [
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
        )

        test_loader = DataLoader(test_set_normal, batch_size=50, num_workers=8)

        return test_loader


stn1_params = {
    "conv1_in": 3,
    "conv1_out": 250,
    "conv2_in": 250,
    "conv2_out": 250,
    "lin1_in": 250 * 6 * 6,
    "lin1_out": 250,
    "lin2_in": 250,
    "lin2_out": 6,
}

stn2_params = {
    "conv1_in": 200,
    "conv1_out": 150,
    "conv2_in": 150,
    "conv2_out": 200,
    "lin1_in": 200 * 2 * 2,
    "lin1_out": 300,
    "lin2_in": 300,
    "lin2_out": 6,
}

stn3_params = {
    "conv1_in": 250,
    "conv1_out": 150,
    "conv2_in": 150,
    "conv2_out": 200,
    "lin1_in": 200 * 1 * 1,
    "lin1_out": 300,
    "lin2_in": 300,
    "lin2_out": 6,
}

stn_params = [stn1_params, stn2_params, stn3_params]

hparams = {
    "dropout_rate": 0.45,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "optimizer": "SGD",
    "activation_function_features": "RELU",
    "activation_function_classifier": "RELU",
    "stn_parameter": stn_params,
}
