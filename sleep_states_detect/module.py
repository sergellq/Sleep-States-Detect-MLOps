import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy


class SleepStateClassifier(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_acc = Accuracy(task="multiclass", num_classes=5)
        self.val_acc = Accuracy(task="multiclass", num_classes=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
