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


# class MultiModalCNN(nn.Module):
#     def __init__(self, num_classes=1, fusion='it', seq_length=15, pretr_ef='None', num_heads=1):
#         super(MultiModalCNN, self).__init__()
#         assert fusion in ['ia', 'it', 'lt'], print('Unsupported fusion method: {}'.format(fusion))

#         # self.audio_model = AudioCNNPool(num_classes=num_classes)
#         # self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)

#         init_feature_extractor(self.visual_model, pretr_ef)
                           
#         e_dim = 128
#         input_dim_video = 128
#         input_dim_audio = 128
#         input_dim_video = 768 #marlin size
#         # input_dim_audio = 128
#         self.fusion=fusion

#         if fusion in ['lt', 'it']:
#             if fusion  == 'lt':
#                 self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
#                 self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
#             elif fusion == 'it': # using this one 
#                 # input_dim_video = input_dim_video // 2
#                 input_dim_video = input_dim_video # why was there a //2? is it because conv layers diff?
#                 self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
#                 self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)   
        
#         elif fusion in ['ia']:
#             input_dim_video = input_dim_video // 2
            
#             self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
#             self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)

            
#         self.classifier_1 = nn.Sequential(
#                     nn.Linear(e_dim*2, num_classes), # depfake so 1
#                 )
        
            

#     def forward(self, x_audio, x_visual):

#         if self.fusion == 'lt':
#             return self.forward_transformer(x_audio, x_visual)

#         elif self.fusion == 'ia':
#             return self.forward_feature_2(x_audio, x_visual)
       
#         elif self.fusion == 'it': # concerend with this
#             return self.forward_feature_3(x_audio, x_visual)

 
        
#     def forward_feature_3(self, x_audio, x_visual):
#         x_audio = self.audio_model.forward_stage1(x_audio)
#         x_visual = self.visual_model.forward_features(x_visual)
#         x_visual = self.visual_model.forward_stage1(x_visual)

#         proj_x_a = x_audio.permute(0,2,1)
#         proj_x_v = x_visual.permute(0,2,1)

#         h_av = self.av1(proj_x_v, proj_x_a)
#         h_va = self.va1(proj_x_a, proj_x_v)
        
#         h_av = h_av.permute(0,2,1)
#         h_va = h_va.permute(0,2,1)
        
#         x_audio = h_av+x_audio
#         x_visual = h_va + x_visual

#         x_audio = self.audio_model.forward_stage2(x_audio)       
#         x_visual = self.visual_model.forward_stage2(x_visual)
        
#         audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
#         video_pooled = x_visual.mean([-1])

#         x = torch.cat((audio_pooled, video_pooled), dim=-1)
#         x1 = self.classifier_1(x)
#         return x1
    
#     def forward_feature_2(self, x_audio, x_visual):
#         x_audio = self.audio_model.forward_stage1(x_audio)
#         x_visual = self.visual_model.forward_features(x_visual)
#         x_visual = self.visual_model.forward_stage1(x_visual)

#         proj_x_a = x_audio.permute(0,2,1)
#         proj_x_v = x_visual.permute(0,2,1)

#         _, h_av = self.av1(proj_x_v, proj_x_a)
#         _, h_va = self.va1(proj_x_a, proj_x_v)
        
#         if h_av.size(1) > 1: #if more than 1 head, take average
#             h_av = torch.mean(h_av, axis=1).unsqueeze(1)
       
#         h_av = h_av.sum([-2])

#         if h_va.size(1) > 1: #if more than 1 head, take average
#             h_va = torch.mean(h_va, axis=1).unsqueeze(1)

#         h_va = h_va.sum([-2])

#         x_audio = h_va*x_audio
#         x_visual = h_av*x_visual
        
#         x_audio = self.audio_model.forward_stage2(x_audio)       
#         x_visual = self.visual_model.forward_stage2(x_visual)

#         audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
#         video_pooled = x_visual.mean([-1])
        
#         x = torch.cat((audio_pooled, video_pooled), dim=-1)
        
#         x1 = self.classifier_1(x)
#         return x1

#     def forward_transformer(self, x_audio, x_visual):
#         x_audio = self.audio_model.forward_stage1(x_audio)
#         proj_x_a = self.audio_model.forward_stage2(x_audio)
       
#         x_visual = self.visual_model.forward_features(x_visual) 
#         x_visual = self.visual_model.forward_stage1(x_visual)
#         proj_x_v = self.visual_model.forward_stage2(x_visual)
           
#         proj_x_a = proj_x_a.permute(0, 2, 1)
#         proj_x_v = proj_x_v.permute(0, 2, 1)
#         h_av = self.av(proj_x_v, proj_x_a)
#         h_va = self.va(proj_x_a, proj_x_v)
       
#         audio_pooled = h_av.mean([1]) #mean accross temporal dimension
#         video_pooled = h_va.mean([1])

#         x = torch.cat((audio_pooled, video_pooled), dim=-1)  
#         x1 = self.classifier_1(x)
#         return x1
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    