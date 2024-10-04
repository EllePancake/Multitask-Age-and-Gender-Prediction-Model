import torch
from torch import nn
import torchvision.models as models
import pytorch_lightning as pl


class MultiTaskResNet(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.learning_rate = learning_rate
        self.criterion_age = nn.MSELoss()
        self.criterion_gender = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.resnet(x)

        age = self.age_head(features)
        gender = self.gender_head(features)

        return age, gender

    def training_step(self, batch, batch_idx):
        images, targets = batch
        age_targets = targets[:, 0].float()
        gender_targets = targets[:, 1].long()

        age_preds, gender_preds = self(images)

        age_loss = self.criterion_age(age_preds.squeeze(), age_targets)
        gender_loss = self.criterion_gender(gender_preds, gender_targets)

        total_loss = age_loss + gender_loss

        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_age_loss', age_loss, prog_bar=True)
        self.log('train_gender_loss', gender_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        age_targets = targets[:, 0].float()
        gender_targets = targets[:, 1].long()

        age_preds, gender_preds = self(images)

        age_loss = self.criterion_age(age_preds.squeeze(), age_targets)
        gender_loss = self.criterion_gender(gender_preds, gender_targets)

        total_loss = age_loss + gender_loss

        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_age_loss', age_loss, prog_bar=True)
        self.log('val_gender_loss', gender_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
