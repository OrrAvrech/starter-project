import torch
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

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x[0])
        predictions = self.fc(representations)
        return predictions

    def forward_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_step(batch)
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
