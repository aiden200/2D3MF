import librosa
import numpy as np
import torch.nn.functional as F
from torch import nn

from src.marlin_pytorch.util import audio_load, get_mfccs


def get_audio_features(file_path, min_duration=3):
    # Load the audio file
    audio, sr = audio_load(file_path)  # 3 seconds

    # Padding or Cutting by min_duration
    audio_length = min_duration * sr
    if audio.shape[0] < min_duration * sr:
        # Padding
        audio = np.pad(audio, (0, audio_length - len(audio)), 'constant')
    elif audio.shape[0] > audio_length:
        # Cutting
        audio = audio[:audio_length]

    audio_frames = librosa.util.frame(audio, frame_length=sr, hop_length=sr)
    audio_frames = np.transpose(audio_frames)
    assert audio_frames.shape[0] == min_duration, f"The audio frames should have {min_duration} seconds duration."

    # Extract audio features
    audio_features = np.array([get_mfccs(y=audio_frame, sr=sr) for audio_frame in audio_frames])  # (3, 10, 87)
    return audio_features


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class AudioResNet18(nn.Module):
    """
        This model is a training model for audio features with ResNet18
    """

    def __init__(self, num_classes=8):
        super(AudioResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)  # It is not used for extracting features.

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)  # (B, 512)
        # out = self.fc(out)  # When we train the model, it should be active
        return out
