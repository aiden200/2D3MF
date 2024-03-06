from typing import Optional, Union, Sequence, Dict, Literal, Any

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Identity, BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torch.nn import BatchNorm1d, LayerNorm, ReLU, LeakyReLU
from .transformer_blocks import AttentionBlock

import torch.nn as nn
import time
import torch

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 

class Classifier(LightningModule):

    def __init__(self, num_classes: int, backbone: str, finetune: bool,
        marlin_ckpt: Optional[str] = None,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        learning_rate: float = 1e-4, distributed: bool = False,
        ir_layers = "conv",
        num_heads = 1
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

        config = resolve_config(backbone)


        self.hidden_layers = 128
        self.input_dim_audio = 128 # placeholder

        if ir_layers == "fc":
            self.layer_norm = LayerNorm(config.encoder_embed_dim)
            self.fc = Linear(config.encoder_embed_dim, self.hidden_layers)
            self.layer_norm2 = LayerNorm(self.hidden_layers)
            # self.fc2 = Linear(self.hidden_layers, num_classes)
        elif ir_layers == "conv":
            self.conv1d_0 = conv1d_block(config.encoder_embed_dim, 64) #might be too big
            self.conv1d_1 = conv1d_block(64, 64)
            self.conv1d_2 = conv1d_block(64, 128)
            self.conv1d_3 = conv1d_block(128, self.hidden_layers)
        else:
            self.hidden_layers = config.encoder_embed_dim #768

        self.av1 = AttentionBlock(
            in_dim_k=self.hidden_layers, 
            in_dim_q=self.input_dim_audio, 
            out_dim=self.input_dim_audio, 
            num_heads=num_heads)
        self.va1 = AttentionBlock(
            in_dim_k=self.input_dim_audio, 
            in_dim_q=self.hidden_layers, 
            out_dim=self.hidden_layers, 
            num_heads=num_heads)   
        

        self.hidden_layers_2 = 128

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_layers_2, num_classes), # depfake so 1
        )



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

    def extract_audio_features(x_audio):
        pass

    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)

    def forward(self, x_vid):#, x_audio):
        if self.model is not None:
            feat = self.model.extract_features(x_vid, True)
        else:
            feat = x_vid
        
        x_audio = self.extract_audio_features(x_audio)
        x_vid = self.conv1d_0(x_vid) 
        x_vid = self.conv1d_1(x_vid)
        #x_audio = 

        h_av = self.av1(x_vid, x_audio)
        h_va = self.va1(x_audio, x_vid)

        h_av = h_av.permute(0,2,1)
        h_va = h_va.permute(0,2,1)

        x_audio = h_av+x_audio
        x_visual = h_va + x_visual

        x_vid = self.conv1d_2(x_vid) 
        x_vid = self.conv1d_3(x_vid)

        audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
        video_pooled = x_visual.mean([-1])
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x1 = self.classifier_1(x)
        #x_audio = 
        return x1.sigmoid()

        return feat.sigmoid()

    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        # x, y, z = batch # video frames, label, audio mfccs
        x, y= batch # video frames, label, audio mfccs
        y_hat = self(x)
        if self.task == "multilabel":
            y_hat = y_hat.flatten()
            y = y.flatten()
        # print(y_hat, 1 - y.float())
        # print(y_hat, y.float())
        loss = self.loss_fn(y_hat, y.float())
        # print(loss)
        prob = y_hat
        # prob = y_hat[y_hat < 0.5]

        # time.sleep(2)

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
