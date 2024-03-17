import os
from abc import ABC, abstractmethod
from itertools import islice
from typing import Optional

import ffmpeg
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from marlin_pytorch.util import read_video, padding_video
from util.misc import sample_indexes, read_text, read_json

from dataset.utils import *

class CelebvHqBase(LightningDataModule, ABC):

    def __init__(self, data_root: str, split: str, task: str, data_ratio: float = 1.0, take_num: int = None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert task in ("appearance", "action", "deepfake")
        self.task = task
        self.take_num = take_num

        self.name_list = list(
            filter(lambda x: x != "", read_text(os.path.join(data_root, f"{self.split}.txt")).split("\n")))
        # self.metadata = read_json(os.path.join(data_root, "celebvhq_info.json"))

        if data_ratio < 1.0:
            self.name_list = self.name_list[:int(len(self.name_list) * data_ratio)]
        if take_num is not None:
            self.name_list = self.name_list[:self.take_num]

        print(f"Dataset {self.split} has {len(self.name_list)} videos")

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    def __len__(self):
        return len(self.name_list)


# for fine-tuning
class CelebvHq(CelebvHqBase):

    def __init__(self,
        root_dir: str,
        split: str,
        task: str,
        clip_frames: int,
        temporal_sample_rate: int,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None,
        temporal_axis: int = 1
    ):
        super().__init__(root_dir, split, task, data_ratio, take_num)
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.temporal_axis = temporal_axis

    def __getitem__(self, index: int):
        # y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task]
        y = int(self.name_list[index].split("-")[1]) # should be 0-real, 1-fake        
        video_path = os.path.join(self.data_root, "cropped", self.name_list[index] + ".mp4")
        audio_path = os.path.join(self.data_root, "audio", extract_number(self.name_list[index]) + ".mp3")
        probe = ffmpeg.probe(video_path)["streams"][0]
        n_frames = int(probe["nb_frames"])
        
        ## this is for double the time

        temporal_frames = self.clip_frames*self.temporal_axis

        if n_frames <= temporal_frames: # not needed (as long as our videos are > 0.5sec)
            video = read_video(video_path, channel_first=True).video / 255
            # pad frames to 16
            video = padding_video(video, temporal_frames, "same")  # (T, C, H, W)
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
            return video, torch.tensor(y, dtype=torch.long)
        elif n_frames <= temporal_frames * self.temporal_sample_rate: # not needed (as long as our videos are > 1sec)
            # reset a lower temporal sample rate
            sample_rate = n_frames // temporal_frames
        else:
            sample_rate = self.temporal_sample_rate
        # sample frames
            
        ## clip_frames hyperparameters
        
        video_indexes = sample_indexes(n_frames, temporal_frames, sample_rate)
        reader = torchvision.io.VideoReader(video_path)
        fps = reader.get_metadata()["video"]["fps"][0]
        reader.seek(video_indexes[0].item() / fps, True)
        frames = []
        for frame in islice(reader, 0, temporal_frames * sample_rate, sample_rate):
            frames.append(frame["data"])

        video = torch.stack(frames) / 255  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        # clip_frames = how many frames
        
        assert video.shape[1] == temporal_frames, video_path
        
        audio, sr = audio_load(audio_path) # audio has been resampled to 44100 Hz
        start_audio_idx = int((video_indexes[0]/30)*fps) # end_idx -> int((video_indexes[-1]/30)*sr)
        audio = audio[start_audio_idx:start_audio_idx+sr*self.temporal_axis]
        audio_mfccs = self.get_mfccs(audio, sr)
        # print(f"Video shape: {video.shape}")
        return video, torch.tensor([y], dtype=torch.float).bool(), torch.tensor(audio_mfccs) # here we need to return the audio features too

    def get_mfccs(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        return mfcc


# For linear probing
class CelebvHqFeatures(CelebvHqBase):

    def __init__(self, root_dir: str,
        feature_dir: str,
        split: str,
        task: str,
        temporal_reduction: str,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None,
        temporal_axis: int = 1
    ):
        super().__init__(root_dir, split, task, data_ratio, take_num)
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.temporal_axis = temporal_axis

    def __getitem__(self, index: int):
        feat_path = os.path.join(self.data_root, self.feature_dir, self.name_list[index] + ".npy")
        audio_path = os.path.join(self.data_root, "audio_features", extract_number(self.name_list[index]) + ".npy")
        
        x_v = torch.from_numpy(np.load(feat_path)).float()
        x_a = torch.from_numpy(np.load(audio_path))
        # trim or add padding to add up to self.temporal_axis embeddings (~average video duration)
        print("shape of x_v", x_a.dim(), x_a.shape, x_v.shape)
        if x_a.dim() == 3:
            if x_v.shape[0] > self.temporal_axis:
                x_v = x_v[:self.temporal_axis]
                x_a = x_a[:self.temporal_axis]
            else:
                n_pad = self.temporal_axis - x_v.shape[0]
                x_v = torch.cat((x_v, torch.zeros(n_pad, x_v.shape[1])), dim=0)
                print("~~~~~ x_a chspa", x_a.shape)
                print(torch.zeros(n_pad, x_a.shape[1], x_a.shape[2]).shape)
                x_a = torch.cat((x_a, torch.zeros(n_pad, x_a.shape[1], x_a.shape[2])), dim=0)
        elif x_a.dim() == 2:
            print("addinf new dimensions to features", x_v.shape, x_a.shape)
            n_pad = self.temporal_axis
            x_v = torch.cat((x_v, torch.zeros(n_pad, x_v.shape[1])), dim=0)
            x_a = torch.cat((x_a.unsqueeze(0), torch.zeros(n_pad, x_a.shape[0], x_a.shape[1])), dim=0)
        else:
            print("Error: audio features are ill shaped")
        y = int(self.name_list[index].split("-")[1]) # should be 0-real, 1-fake

        print("shape of", x_v.shape, x_a.shape)
        return x_v, torch.tensor([y], dtype=torch.float).bool(), x_a


class CelebvHqDataModule(LightningDataModule):

    def __init__(self, root_dir: str,
        load_raw: bool,
        task: str,
        batch_size: int,
        num_workers: int = 0,
        clip_frames: int = None,
        temporal_sample_rate: int = None,
        feature_dir: str = None,
        temporal_reduction: str = "mean",
        data_ratio: float = 1.0,
        take_train: Optional[int] = None,
        take_val: Optional[int] = None,
        take_test: Optional[int] = None,
        temporal_axis: float = 1.0
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.load_raw = load_raw
        self.data_ratio = data_ratio
        self.take_train = take_train
        self.take_val = take_val
        self.take_test = take_test
        self.temporal_axis = temporal_axis

        if load_raw:
            assert clip_frames is not None
            assert temporal_sample_rate is not None
        else:
            assert feature_dir is not None
            assert temporal_reduction is not None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.load_raw:
            self.train_dataset = CelebvHq(self.root_dir, "train", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_train, temporal_axis=self.temporal_axis)
            self.val_dataset = CelebvHq(self.root_dir, "val", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_val, temporal_axis=self.temporal_axis)
            self.test_dataset = CelebvHq(self.root_dir, "test", self.task, self.clip_frames,
                self.temporal_sample_rate, 1.0, self.take_test, temporal_axis=self.temporal_axis)
        else:
            self.train_dataset = CelebvHqFeatures(self.root_dir, self.feature_dir, "train", self.task,
                self.temporal_reduction, self.data_ratio, self.take_train)
            self.val_dataset = CelebvHqFeatures(self.root_dir, self.feature_dir, "val", self.task,
                self.temporal_reduction, self.data_ratio, self.take_val)
            self.test_dataset = CelebvHqFeatures(self.root_dir, self.feature_dir, "test", self.task,
                self.temporal_reduction, 1.0, self.take_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
