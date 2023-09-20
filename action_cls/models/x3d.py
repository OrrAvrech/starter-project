import torch
from torchmetrics import Accuracy
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LitX3DTransfer(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.model_name = model_name
        model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)
        layers = list(model.blocks.children())
        # feature extractor
        backbone = layers[:-1]
        self.feature_extractor = nn.Sequential(*backbone)
        # classifier
        self.fc = layers[-1]
        num_filters = self.fc.proj.in_features
        self.num_classes = num_classes
        self.fc.proj = nn.Linear(in_features=num_filters, out_features=num_classes, bias=True)
        # metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        representations = self.feature_extractor(x)
        predictions = self.fc(representations)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.feature_extractor.eval()
        with torch.no_grad():
          x = self.feature_extractor(x)
        y_hat = self.fc(x)
        loss = F.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.log("train_acc", self.accuracy, prog_bar=True)
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.log("val_acc", self.accuracy, prog_bar=True)
        self.log("val_loss", loss.mean(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
