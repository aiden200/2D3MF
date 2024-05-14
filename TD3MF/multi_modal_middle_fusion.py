# -*- coding: utf-8 -*-
"""
Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py

Credit to 
@article{chumachenko2022self,
  title={Self-attention fusion for audiovisual emotion recognition with incomplete data},
  author={Chumachenko, Kateryna and Iosifidis, Alexandros and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2201.11095},
  year={2022}
}


https://github.com/katerynaCh/multimodal-emotion-recognition
"""

import torch.nn as nn

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

# Audio CNN (audio embedding)
class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8,h_dim=128,out_dim=128):
        super(AudioCNNPool, self).__init__()

        input_channels = 10
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, h_dim)
        self.conv1d_2 = conv1d_block_audio(h_dim, 256)
        self.conv1d_3 = conv1d_block_audio(256, out_dim)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(h_dim, num_classes),
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


# Audio CNN (audio embedding)
class VideoCnnPool(nn.Module):

    def __init__(self, num_classes=8, input_dim=768, h_dim = 128, out_dim=128):
        super(VideoCnnPool, self).__init__()

        self.conv1d_0 = conv1d_block(input_dim, 64) #might be too big
        self.conv1d_1 = conv1d_block(64, h_dim)
        self.conv1d_2 = conv1d_block(h_dim, 128)
        self.conv1d_3 = conv1d_block(128, out_dim)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(h_dim, num_classes),
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


class EatConvBlock(nn.Module):
    def __init__(self, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=400, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=400, out_channels=768, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        self.downsample = downsample

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, 512, 768) to (B, 768, 512)
        x = self.conv1(x)
        x = self.conv2(x) # (B, 768, 128)
        if self.downsample:
            x = self.adaptive_pool(x) 
        x = x.permute(0, 2, 1) # (B, 128, 768)
        return x

def conv1d_output_size(input_length, kernel_size, stride=1, padding=0):
    output_length = (input_length - kernel_size + 2 * padding) // stride + 1
    return output_length

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    