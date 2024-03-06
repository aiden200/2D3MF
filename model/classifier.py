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
from model.transformer_blocks import AttentionBlock
from model.multi_modal_middle_fusion import AudioCNNPool,VideoCnnPool

import torch.nn as nn
import time
import torch

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config



def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 

def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

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
        print(config.encoder_embed_dim)

        self.hidden_layers = 128
        self.hidden_layers_audio = 128 # placeholder
        self.out_dim = 128

        self.audio_model_cnn = AudioCNNPool(num_classes=1, 
                                            h_dim=self.hidden_layers_audio,
                                            out_dim=self.out_dim)
        self.video_model_cnn = VideoCnnPool(num_classes=1, 
                                            input_dim=config.encoder_embed_dim, 
                                            h_dim=self.hidden_layers,
                                            out_dim=self.out_dim)


        if ir_layers == "fc":
            self.layer_norm = LayerNorm(config.encoder_embed_dim)
            self.fc = Linear(config.encoder_embed_dim, self.hidden_layers)
            self.layer_norm2 = LayerNorm(self.hidden_layers)
            # self.fc2 = Linear(self.hidden_layers, num_classes)


        self.av1 = AttentionBlock(
            in_dim_k=self.hidden_layers, 
            in_dim_q=self.hidden_layers_audio, 
            out_dim=self.hidden_layers_audio, 
            num_heads=num_heads)
        self.va1 = AttentionBlock(
            in_dim_k=self.hidden_layers_audio, 
            in_dim_q=self.hidden_layers, 
            out_dim=self.hidden_layers, 
            num_heads=num_heads)   
        
        self.classifier_1 = nn.Sequential(
                    nn.Linear(self.hidden_layers*2, num_classes), # depfake so 1
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


    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)


    def forward_stage1(self, x_v,x_a):
        x_vid = self.conv1d_0(x_v) 
        x_vid = self.conv1d_1(x_vid)


    def forward(self, x_v, x_a):
        if self.model is not None:
            x_v = self.model.extract_features(x_v, True)
        else:
            x_v = x_v

        print("shape of embedding:", x_a.shape)
        
        x_v = self.video_model_cnn.forward_stage1(x_v)
        x_a = self.audio_model_cnn.forward_stage1(x_a)

        h_av = self.av1(x_v, x_a)
        h_va = self.va1(x_a, x_v)

        h_av = h_av.permute(0,2,1)
        h_va = h_va.permute(0,2,1)

        x_a = h_av + x_a
        x_v = h_va + x_v

        x_v = self.video_model_cnn.forward_stage2(x_v)
        x_a = self.audio_model_cnn.forward_stage2(x_a)

        video_pooled = x_v.mean([-1]) #mean accross temporal dimension
        audio_pooled = x_a.mean([-1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        x1 = self.classifier_1(x)

        return x1.sigmoid()

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



#from torchsummary import summary
#model = AudioCNNPool().to('cuda:0')
#dummy_input = torch.randn((1, 10, 87)).to('cuda:0')
#xx = model(dummy_input)
#print("output shape", xx.shape)
#summary(model, (10, 87))

