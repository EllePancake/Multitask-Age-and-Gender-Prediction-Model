import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from utils_metrics import log_metrics
import torch


class MultiTaskTrainer(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=0.001):
        """
        Initializes the multi-task trainer for training, validation, and
        testing.

        Args:
            model (nn.Module): The model to be trained.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader): DataLoader for the test set.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(MultiTaskTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        age_targets = targets['age'].float()
        gender_targets = targets['gender'].long()

        age_preds, gender_preds = self.model(images)

        age_loss = self.model.criterion_age(age_preds.squeeze(), age_targets)
        gender_loss = self.model.criterion_gender(gender_preds, gender_targets)
        total_loss = age_loss + gender_loss

        log_metrics(self, gender_preds, gender_targets, age_preds,
                    age_targets, stage='train')

        self.log('train_loss', total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        age_targets = targets['age'].float()
        gender_targets = targets['gender'].long()

        age_preds, gender_preds = self.model(images)

        age_loss = self.model.criterion_age(age_preds.squeeze(), age_targets)
        gender_loss = self.model.criterion_gender(gender_preds, gender_targets)
        total_loss = age_loss + gender_loss

        log_metrics(self, gender_preds, gender_targets, age_preds,
                    age_targets, stage='val')

        self.log('val_loss', total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)
        return optimizer


def train_model(model, train_loader, val_loader, test_loader, version,
                max_epochs=10, checkpoint_dir='models/checkpoints/'):
    """
    Function to train a multi-task model using PyTorch Lightning.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        version (str): The version of the model (e.g., 'v1', 'v2', etc.).
        max_epochs (int): Maximum number of epochs for training.
        checkpoint_dir (str): Directory to save the model checkpoints.
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{version}_best',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    task_trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=model.learning_rate
    )

    trainer.fit(task_trainer, train_loader, val_loader)

    metrics_dict = {
        'version': version,
        'avg_train_loss': trainer.logged_metrics.get('train_loss', None),
        'avg_val_loss': trainer.logged_metrics.get('val_loss', None),
        'train_gender_accuracy': trainer.logged_metrics.get(
            'train_gender_accuracy', None),
        'train_age_mae': trainer.logged_metrics.get('train_age_mae', None),
        'val_gender_accuracy': trainer.logged_metrics.get(
            'val_gender_accuracy', None),
        'val_age_mae': trainer.logged_metrics.get('val_age_mae', None)
    }

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv('results/metrics.csv',
                      mode='a',
                      header=False,
                      index=False
                      )

    print(f"Training complete for {version}. Checkpoint saved to")
