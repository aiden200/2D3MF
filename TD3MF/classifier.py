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
from TD3MF.transformer_blocks import AttentionBlock, PositionalEncoding
from TD3MF.multi_modal_middle_fusion import AudioCNNPool, VideoCnnPool, EatConvBlock
from TD3MF.extract_pretrained_features import forward_audio_model, forward_video_model
from moviepy.editor import VideoFileClip
import os


import torch.nn as nn
import time
import torch
import numpy as np

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True))


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'), nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'), nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


class TD3MF(LightningModule):

    def __init__(self, num_classes: int, backbone: str, finetune: bool,
                 marlin_ckpt: Optional[str] = None,
                 task: Literal["binary", "multiclass",
                               "multilabel"] = "binary",
                 learning_rate: float = 1e-4, distributed: bool = False,
                 ir_layers="conv",
                 num_heads=1,
                 temporal_axis: int = 1,
                 audio_pe: bool = True,
                 fusion: str = "lf",
                 hidden_layers: int = 128,
                 lp_only: bool = False,
                 audio_backbone: str = "MFCC",
                 middle_fusion_type: str = "default",
                 video_backbone: str = "marlin"
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

        self.marlin_backbone = backbone
        self.video_backbone = video_backbone
        self.temporal_axis = temporal_axis
        self.hidden_layers = hidden_layers
        self.audio_backbone = audio_backbone
        self.middle_fusion_type = middle_fusion_type
        self.out_dim = self.hidden_layers
        self.audio_pe = None
        self.lf = False
        if fusion == "lf":
            self.lf = True


        if video_backbone == "efficientface":
            config.encoder_embed_dim = 1024 


        if audio_backbone == "MFCC":
            self.audio_hidden_layers = self.hidden_layers
        elif audio_backbone == "eat":
            self.audio_hidden_layers = 768
            downsample = False
            self.eat_down = EatConvBlock(downsample) # brings (B, 512, 768) -> (B, 128, 768). Downsample brings it to (B, 10, 768)
        elif audio_backbone == "xvectors":
            self.audio_hidden_layers = 768
            self.fc_xvec = nn.Linear(7205, self.audio_hidden_layers) # project to a smaller dimension
        elif audio_backbone == "emotion2vec":
            self.audio_hidden_layers = 768
        elif audio_backbone == "resnet":
            self.audio_hidden_layers = 512
        else:
            raise ValueError("Unsupported audio backbone: Must be one of (MFCC, eat, xvectors, emotion2vec, resnet)")
        # add resenet and stuff

        self.lp_only = lp_only
        if lp_only:
            self.lp_only_fc = nn.Linear(
                self.temporal_axis * config.encoder_embed_dim, num_classes)

        self.audio_model_cnn = AudioCNNPool(num_classes=128,
                                            h_dim=self.audio_hidden_layers,  # audio hidden layers
                                            out_dim=self.out_dim)
        self.video_model_cnn = VideoCnnPool(num_classes=1,
                                            input_dim=config.encoder_embed_dim,
                                            h_dim=self.hidden_layers,
                                            out_dim=self.out_dim)

        self.project_down = nn.Linear(
            config.encoder_embed_dim, self.hidden_layers)

        if ir_layers == "fc":
            self.layer_norm = LayerNorm(config.encoder_embed_dim)
            self.fc = Linear(config.encoder_embed_dim, self.hidden_layers)
            self.layer_norm2 = LayerNorm(self.hidden_layers)
            # self.fc2 = Linear(self.hidden_layers, num_classes)

        if audio_pe:
            self.audio_pe = PositionalEncoding(
                d_model=self.audio_hidden_layers,  # audio hidden layers
                dropout=0.1,
                max_len=self.temporal_axis)
            if audio_backbone == "eat":
                self.audio_pe = PositionalEncoding(
                    d_model=self.audio_hidden_layers,  # audio hidden layers
                    dropout=0.1,
                    max_len=512)
        
        self.av1 = AttentionBlock(
            in_dim_k=self.hidden_layers, 
            in_dim_q=self.audio_hidden_layers, 
            out_dim=self.audio_hidden_layers, 
            num_heads=num_heads)
        self.va1 = AttentionBlock(
            in_dim_k=self.audio_hidden_layers, 
            in_dim_q=self.hidden_layers, 
            out_dim=self.hidden_layers, 
            num_heads=num_heads)  
        if self.middle_fusion_type == 'self_attention' or self.middle_fusion_type == 'self_cross_attention':
            self.vv = AttentionBlock(
                in_dim_k=self.hidden_layers, 
                in_dim_q=self.hidden_layers, 
                out_dim=self.hidden_layers, 
                num_heads=num_heads)
            self.aa = AttentionBlock(
                in_dim_k=self.audio_hidden_layers, 
                in_dim_q=self.audio_hidden_layers, 
                out_dim=self.audio_hidden_layers, 
                num_heads=num_heads)
        self.av2 = AttentionBlock(
            in_dim_k=self.audio_hidden_layers, #audio hidden layers
            in_dim_q=self.hidden_layers,
            out_dim=self.audio_hidden_layers, #audio hidden layers
            num_heads=num_heads)
        self.va2 = AttentionBlock(
            in_dim_k=self.hidden_layers,
            in_dim_q=self.audio_hidden_layers, #audio hidden layers
            out_dim=self.hidden_layers,
            num_heads=num_heads) 

        self.classifier_1 = nn.Sequential(
            nn.Linear(self.hidden_layers*2, num_classes),  # depfake so 1
        )
        if self.lf:
            self.classifier_1 = nn.Sequential(
                nn.Linear(self.hidden_layers + self.audio_hidden_layers, num_classes),  # depfake so 1
            )

        self.learning_rate = learning_rate
        self.distributed = distributed
        self.task = task

        if task == "binary":
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

        print(f"{'-'*30}\nHyperparameters:\n{'-'*30}\nModel: {backbone}\nFinetune: {finetune}\nTask:\
{task}\nLearning Rate: {learning_rate}\nDistributed: {distributed}\n\
IR Layers: {ir_layers}\nNum Heads: {num_heads}\nTemporal Axis: {temporal_axis}\n\
Audio Positional Encoding: {audio_pe}\nFusion: {fusion}\nHidden layer size: {self.hidden_layers}\n\
lp_only: {lp_only}\nAudio Backbone: {audio_backbone}\n{'-'*30}")

    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)

    def forward(self, x_v, x_a):
        # print(x_v.shape, x_a.shape)

        if self.model is not None:  # MARLIN ft
            # (B, temporal, T, C, H, W)
            x_v = x_v.permute(0, 1, 3, 2, 4, 5)
            # Encoder takes in (C, T, H, W)
            x_v_split = x_v.view(
                (x_v.shape[1] * x_v.shape[0], x_v.shape[2], x_v.shape[3], x_v.shape[4], x_v.shape[5]))
            x_v = self.model.extract_features(x_v_split, True)
            # now we need to seperate it back to normal (B, E) -> (B,T,E)
            x_v = x_v.reshape(
                (x_v.shape[0]//self.temporal_axis, self.temporal_axis, x_v.shape[-1]))

        x = self._extract(x_v, x_a)

        x1 = self.classifier_1(x)

        return x1.sigmoid()
    
    def _extract(self, x_v, x_a):
        if self.lp_only:  # only linear probing
            return self.lp_only_fc(x_v)

        if self.audio_backbone == "MFCC":
            x_a = x_a.view(
                (x_a.shape[0]*self.temporal_axis, x_a.shape[2], x_a.shape[3]))
            x_a = self.audio_model_cnn.forward(x_a)
            x_a = x_a.view(
                (x_a.shape[0]//self.temporal_axis, self.temporal_axis, x_a.shape[1]))
        elif self.audio_backbone == "xvectors":
            x_a = self.fc_xvec(x_a) # project embedding 7205 -> 128  
        elif self.audio_backbone == "eat":
            x_a = self.eat_down(x_a) # (B, 512, 768) -> (B, 128, 768) 


        if self.audio_pe:
            # (B, T, E)
            x_a = x_a.permute(1, 0, 2)
            x_a = self.audio_pe(x_a)
            x_a = x_a.permute(1, 0, 2)


        x_v = self.project_down(x_v)

        if self.middle_fusion_type == 'default':
            # DEFAULT MIDDLE FUSION
            h_av = self.av1(x_v, x_a)  
            h_va = self.va1(x_a, x_v)  
        elif self.middle_fusion_type == 'audio_refuse': 
            h_va = self.va1(x_a, x_v)
            h_av = self.av1(x_v*h_va, x_a)
        elif self.middle_fusion_type == 'video_refuse':
            h_av = self.av1(x_v, x_a)  
            h_va = self.va1(x_a*h_av, x_v) 
        elif self.middle_fusion_type == 'self_attention':
            h_va = self.vv(x_v, x_v) 
            h_av = self.aa(x_a, x_a) 
        elif self.middle_fusion_type == 'multi_attention':
            h_av1 = self.av1(x_v, x_a)  
            h_va1 = self.va1(x_a, x_v)  
            x_a = x_a * h_av1
            x_v = x_v * h_va1
            h_av2 = self.av1(x_v, x_a)  
            h_va2 = self.va1(x_a, x_v)
            h_av = h_av2
            h_va = h_va2
        elif self.middle_fusion_type == "self_cross_attention":
            h_vv = self.vv(x_v, x_v) 
            h_aa = self.aa(x_a, x_a)  
            x_v = x_v * h_vv
            x_a = x_a * h_aa
            h_av = self.av1(x_v, x_a)  
            h_va = self.va1(x_a, x_v) 
        else:
            raise ValueError(f"Incorrect middle fusion type: {self.middle_fusion_type}\nMust be one of (default, alternate1, self-attention, cross-attention)")
        
        
        x_a = h_av * x_a
        x_v = h_va * x_v

        x_v = x_v.permute(0, 2, 1)
        x_a = x_a.permute(0, 2, 1)

        # print(x_v.shape, x_a.shape)

        if not self.lf:
            x_v = self.video_model_cnn.forward_stage2(x_v)
            x_a = self.audio_model_cnn.forward_stage2(x_a)


        video_pooled = x_v.mean([-1])  # mean accross temporal dimension
        audio_pooled = x_a.mean([-1])
        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        return x

    
    def feature_extraction(self, file_path):
        if not os.path.exists("temp"):
            os.mkdir("temp")
        audio_output_path = os.path.join("temp", "audio_clip.wav")
        video_output_path = os.path.join("temp", "video_clip.mp4")

        clip = VideoFileClip(file_path)
        audio = clip.audio
        audio.write_audiofile(audio_output_path, codec='pcm_s16le')  # Saving the audio as WAV
        video = clip.without_audio()
        video.write_videofile(video_output_path, codec='libx264')  # Saving the video as MP4
        # cut it down to 10 seconds
        audio.close()
        video.close()
        clip.close()

        # run through pretrained models
        if "marlin" in self.video_backbone:
            video_model = self.marlin_backbone
        else:
            video_model = self.video_backbone
        with torch.no_grad():
            x_v = forward_video_model(video_output_path, video_model)
            x_a = forward_audio_model(x_a, self.audio_backbone, x_v)
            # run through pretrained weights
            out = self._extract(x_v, x_a)
            
        return out

    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        x_v, y, x_a = batch  # video frames, label, audio mfccs

        y_hat = self(x_v, x_a)

        if self.task == "multilabel":  # this is incorrect because we throw a sigmoid.
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
        return self(batch[0], batch[2])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate, betas=(0.5, 0.9))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "train_loss"
            }
        }
