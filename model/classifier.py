from typing import Optional, Union, Sequence, Dict, Literal, Any

from pytorch_lightning import LightningModule
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Linear, Identity, BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torch.nn import BatchNorm1d, LayerNorm, ReLU, LeakyReLU

import time

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config

class Classifier(LightningModule):

    def __init__(self, num_classes: int, backbone: str, finetune: bool,
        marlin_ckpt: Optional[str] = None,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        learning_rate: float = 1e-4, distributed: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        if finetune:
            if marlin_ckpt is None:
                self.model = Marlin.from_online(backbone).encoder
            else:
                self.model = Marlin.from_file(backbone, marlin_ckpt).encoder
        else:
            self.model = None

        self.audio_model = AudioCNNPool()
        config = resolve_config(backbone)


        hidden_layers = 128
        self.layer_norm = LayerNorm(config.encoder_embed_dim)
        self.fc = Linear(config.encoder_embed_dim, hidden_layers)
        self.layer_norm2 = LayerNorm(hidden_layers)
        self.fc2 = Linear(hidden_layers, num_classes)

        self.learning_rate = learning_rate
        self.distributed = distributed
        self.task = task
        if task in "binary":
            self.loss_fn = BCELoss()
            self.acc_fn = BinaryAccuracy()
            self.auc_fn = BinaryAUROC()
        elif task == "multiclass":
            self.loss_fn = CrossEntropyLoss()
            self.acc_fn = Accuracy(task=task, num_classes=num_classes)
            self.auc_fn = AUROC(task=task, num_classes=num_classes)
        elif task == "multilabel":
            self.loss_fn = BCELoss()
            self.acc_fn = Accuracy(task="binary", num_classes=1)
            self.auc_fn = AUROC(task="binary", num_classes=1)

    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)

    def forward(self, x_v, x_a):
        if self.model is not None:
            feat_v = self.model.extract_features(x_v, True)
        else:
            feat_v = x_v
        
        # Video branch embeddings
        feat_v = self.layer_norm(feat_v)
        feat_v = self.fc(feat_v)
        feat_v = ReLU()(feat_v)
        feat_v = self.fc2(feat_v)
        # Audio branch embeddings
        feat_a = self.audio_model(x_a)
        print("shape of embedding:", feat_a.shape)

        # Transformer fusion

        return feat_v.sigmoid()

    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        x_v, y, x_a = batch # video frames, label, audio mfccs
        
        y_hat = self(x_v, x_a)
        
        if self.task == "multilabel":
            y_hat = y_hat.flatten()
            y = y.flatten()
        
        loss = self.loss_fn(y_hat, y.float())
        prob = y_hat

        acc = self.acc_fn(prob, y.float())
        auc = self.auc_fn(prob, y.float())
        return {"loss": loss, "acc": acc, "auc": auc}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "train_loss"
            }
        }


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

# Audio CNN (audio embedding)
class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()

        input_channels = 10
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
            
    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x

    def forward_stage1(self,x):            
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
    
    def forward_stage2(self,x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)   
        return x
    
    def forward_classifier(self, x):   
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

#from torchsummary import summary
#model = AudioCNNPool().to('cuda:0')
#dummy_input = torch.randn((1, 10, 87)).to('cuda:0')
#xx = model(dummy_input)
#print("output shape", xx.shape)
#summary(model, (10, 87))

