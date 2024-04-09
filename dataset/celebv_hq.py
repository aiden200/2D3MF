import os
from abc import ABC, abstractmethod
from itertools import islice
from typing import Optional
from collections import deque

import cv2
import ffmpeg
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from marlin_pytorch.util import read_video, padding_video
from util.misc import sample_indexes, read_text, read_json

from dataset.utils import *

class BaseDataSetLoader(LightningDataModule, ABC):

    def __init__(self, data_root: str, split:str, training_datasets: list, eval_datasets: list, task: str, data_ratio: float = 1.0, take_num: int = None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert task in ("appearance", "action", "deepfake")
        self.task = task
        self.take_num = take_num
        self.name_list = []

        if "train" in split:
            for dataset in training_datasets:
                # contain all splits
                dataset_path = os.path.join(data_root, dataset)
                if dataset not in eval_datasets:
                    for dataset_split in [split, split.replace("train", "test")]:
                        path = os.path.join(dataset_path, f"{dataset_split}.txt")
                        assert os.path.exists(path), f"Missing split {path}"
                        files = list(filter(lambda x: x != "", read_text(path).split("\n")))
                        self.name_list += [(dataset_path, x) for x in files]

                else:
                    path = os.path.join(dataset_path, f"{split}.txt")
                    assert os.path.exists(path), f"Missing split {path}"
                    files = list(filter(lambda x: x != "", read_text(path).split("\n")))
                    self.name_list += [(dataset_path, x) for x in files]

        elif "val" in split: # only test datasets are included in val
            for dataset in training_datasets:
                
                dataset_path = os.path.join(data_root, dataset)
                path = os.path.join(dataset_path, f"{split}.txt")
                assert os.path.exists(path), f"Missing split {path}"
                files = list(filter(lambda x: x != "", read_text(path).split("\n")))
                self.name_list += [(dataset_path, x) for x in files]
                
        else:
            for dataset in training_datasets:

                dataset_path = os.path.join(data_root, dataset)
                if dataset not in training_datasets:
                    for dataset_split in [split, split.replace("test", "train"), split.replace("test", "val")]:
                        path = os.path.join(dataset_path, f"{dataset_split}.txt")
                        assert os.path.exists(path), f"Missing split {path}"
                        files = list(filter(lambda x: x != "", read_text(path).split("\n")))
                        self.name_list += [(dataset_path, x) for x in files]
                else:
                    path = os.path.join(dataset_path, f"{split}.txt")
                    assert os.path.exists(path), f"Missing split {path}"
                    files = list(filter(lambda x: x != "", read_text(path).split("\n")))
                    self.name_list += [(dataset_path, x) for x in files]

        

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
class FTDataset(BaseDataSetLoader):

    def __init__(self,
        root_dir: str,
        split: str,
        task: str,
        training_datasets: list,
        eval_datasets: list,
        clip_frames: int,
        temporal_sample_rate: int,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None,
        temporal_axis: int = 1,
        audio_feature: str = "mfcc"
    ):
        super().__init__(root_dir, split, task, data_ratio, take_num)
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.temporal_axis = temporal_axis
        self.audio_feature = audio_feature

    def __getitem__(self, index: int):
        # y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task]
        y = int(self.name_list[index].split("-")[-1]) # should be 0-real, 1-fake        
        video_path = os.path.join(self.data_root, "cropped", self.name_list[index] + ".mp4")
        if self.audio_feature == "mfcc":
            audio_feature_dir = "audio_features"
        elif self.audio_feature == "eat":
            audio_feature_dir = "eat_features"
        ## TODO: implement rest of place & fix audio load
        
        audio_path = os.path.join(self.data_root, audio_feature_dir, self.name_list[index] + ".npy")
        probe = ffmpeg.probe(video_path)["streams"][0]
        n_frames = int(probe["nb_frames"])
        
        video, real_t = self._load_video(video_path, self.temporal_sample_rate, stride=self.clip_frames, temporal=self.temporal_axis)
        #(temporal, T, C, H, W)
        audio_path = os.path.join(self.data_root, "audio_features", self.name_list[index] + ".npy")
        if os.path.exists(audio_path):
            audio = torch.from_numpy(np.load(audio_path))
            if audio.shape[0] > self.temporal_axis:
                audio = audio[:self.temporal_axis]
        else:
            audio = self._load_audio(audio_path, video_path, real_t)
        #(T, C-10, S-87)
        if audio.shape[0] < self.temporal_axis:
            padding = torch.zeros((self.temporal_axis - audio.shape[0], audio.shape[1], audio.shape[2]))
            audio = torch.concatenate((audio, padding), axis=0)

        assert video.shape[0] == self.temporal_axis, f"Video features are not of the right shape {video.shape[0]} != {self.temporal_axis}"
        assert audio.shape[0] == self.temporal_axis, f"Audio features are not of the right shape {audio.shape[0]} != {self.temporal_axis}"
        return video, torch.tensor([y], dtype=torch.float).bool(), audio
        
    def get_mfccs(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        return mfcc
    
    
    def _load_audio(self, audio_path: str, video_path: str, temporal_size: int):
        reader = torchvision.io.VideoReader(video_path)
        fps = reader.get_metadata()["video"]["fps"][0]
        # print(reader.get_metadata()["video"])
        audio, sr = audio_load(audio_path)
        audio_features = []
        # print(audio.shape)
        for i in range(temporal_size):
            start_idx = i * sr
            audio_window = audio[start_idx:start_idx+sr]

            audio_feat = self.get_mfccs(audio_window, sr)
            audio_features.append(audio_feat)
        audio_features = [torch.from_numpy(arr).unsqueeze(0) for arr in audio_features]
        audio_features = torch.cat(audio_features, dim=0)
        return audio_features

    
    def _load_video(self, video_path: str, sample_rate: int, stride: int, temporal:int):
        probe = ffmpeg.probe(video_path)
        total_frames = int(probe["streams"][0]["nb_frames"])
        if total_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True) / 255  # (T, C, H, W)
            # pad frames to 16
            v = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            assert v.shape[0] == self.clip_frames
            return v.permute(1, 0, 2, 3)#.to(self.device)
        elif total_frames <= self.clip_frames * sample_rate:
            video = read_video(video_path, channel_first=True) / 255  # (T, C, H, W)
            # use first 16 frames
            if video.shape[0] < self.clip_frames:
                # double-check the number of frames, see https://github.com/pytorch/vision/issues/2490
                v = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            v = video[:self.clip_frames]
            return v.permute(1, 0, 2, 3)#.to(self.device)
        else:
            # extract features based on sliding window
            cap = cv2.VideoCapture(video_path)
            deq = deque(maxlen=self.clip_frames)

            clip_start_indexes = list(range(0, total_frames - self.clip_frames * sample_rate, stride * sample_rate))
            clip_end_indexes = [i + self.clip_frames * sample_rate - 1 for i in clip_start_indexes]
            current_index = -1
            
            video_clip = torch.zeros((temporal, stride, 3, 224, 224))
            
            index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_index += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1) / 255  # (C, H, W)

                for _ in range(sample_rate - 1):
                    cap.read()
                    current_index += 1

                deq.append(frame)
                if current_index in clip_end_indexes:
                    video_clip[index] = torch.stack(list(deq))  # (T, C, H, W)
                    # return v.permute(1, 0, 2, 3)#.to(self.device)
                    index += 1
                    if index == video_clip.shape[0]:
                        break

            cap.release()
            return video_clip, index # (T, 16, 3, 224, 224)


# For linear probing
class LPFeaturesDataset(BaseDataSetLoader):

    def __init__(self, root_dir: str,
        feature_dir: str,
        split: str,
        training_datasets: list,
        eval_datasets: list,
        task: str,
        temporal_reduction: str,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None,
        temporal_axis: int = 14,
        audio_feature: str = "MFCC"
    ):
        super().__init__(root_dir, split, training_datasets, eval_datasets, task, data_ratio, take_num)
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.temporal_axis = temporal_axis
        self.audio_feature = audio_feature

    def __getitem__(self, index: int):
        feat_path = os.path.join(self.name_list[index][0], self.feature_dir, self.name_list[index][1] + ".npy")

        audio_feature_dir = self.audio_feature
        audio_path = os.path.join(self.name_list[index][0], audio_feature_dir, self.name_list[index][1] + ".npy")
        

        x_v = torch.from_numpy(np.load(feat_path)).float()
        x_a = torch.from_numpy(np.load(audio_path))

        if self.audio_feature == "eat":
            # x_a = x_a.unsqueeze(0)
            if x_v.shape[0] > self.temporal_axis:
                x_v = x_v[:self.temporal_axis]
            else:
                n_pad = self.temporal_axis - x_v.shape[0]
                x_v = torch.cat((x_v, torch.zeros(n_pad, x_v.shape[1])), dim=0)
        elif self.audio_feature == "MFCC":
            if x_a.dim() == 3:
                if x_v.shape[0] > self.temporal_axis:
                    x_v = x_v[:self.temporal_axis]
                    x_a = x_a[:self.temporal_axis]
                else:
                    n_pad = self.temporal_axis - x_v.shape[0]
                    x_v = torch.cat((x_v, torch.zeros(n_pad, x_v.shape[1])), dim=0)
                    x_a = torch.cat((x_a, torch.zeros(n_pad, x_a.shape[1], x_a.shape[2])), dim=0)
            elif x_a.dim() == 2:
                n_pad = self.temporal_axis
                x_v = torch.cat((x_v, torch.zeros(n_pad, x_v.shape[1])), dim=0)
                x_a = torch.cat((x_a.unsqueeze(0), torch.zeros(n_pad, x_a.shape[0], x_a.shape[1])), dim=0)
            else:
                print("Error: audio features are ill shaped")
        elif self.audio_feature == "xvectors":
            #TODO: Implement feature extraction logic
            pass
        elif self.audio_feature == "resnet":
            #TODO: Implement feature extraction logic
            pass
        elif self.audio_feature == "emotion2vec":
            if x_a.dim() == 2:
                if x_a.shape[0] > self.temporal_axis:
                    x_a = x_a[:self.temporal_axis]
                else:
                    n_pad = self.temporal_axis - x_a.shape[0]
                    x_a = torch.cat((x_a, torch.zeros(n_pad, x_a.shape[1])), dim=0)
        else:
            raise ValueError(f"Error in LPFeaturesDataset, incorrect audio backbone: {self.audio_feature}")
        y = int(self.name_list[index][1].split("-")[-1]) # should be 0-real, 1-fake
        
        # print(x_a.shape, x_v.shape)
        return x_v, torch.tensor([y], dtype=torch.float).bool(), x_a


class DataModule(LightningDataModule):

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
        temporal_axis: float = 1.0,
        audio_feature: str = "MFCC",
        training_datasets: list = [],
        eval_datasets: list = []
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
        self.audio_feature = audio_feature

        if load_raw:
            assert clip_frames is not None
            assert temporal_sample_rate is not None
        else:
            assert feature_dir is not None
            assert temporal_reduction is not None
        
        self.training_datasets = training_datasets
        self.eval_datasets = eval_datasets

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.load_raw:
            self.train_dataset = FTDataset(self.root_dir, f"train_{self.audio_feature}", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_train, temporal_axis=self.temporal_axis,audio_feature=self.audio_feature)
            self.val_dataset = FTDataset(self.root_dir, f"val_{self.audio_feature}", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_val, temporal_axis=self.temporal_axis,audio_feature=self.audio_feature)
            self.test_dataset = FTDataset(self.root_dir, f"test_{self.audio_feature}", self.task, self.clip_frames,
                self.temporal_sample_rate, 1.0, self.take_test, temporal_axis=self.temporal_axis,audio_feature=self.audio_feature)
        else:
            self.train_dataset = LPFeaturesDataset(self.root_dir, self.feature_dir, f"train_{self.audio_feature}", self.training_datasets, self.eval_datasets, self.task,
                self.temporal_reduction, self.data_ratio, self.take_train, temporal_axis=self.temporal_axis,audio_feature=self.audio_feature)
            self.val_dataset = LPFeaturesDataset(self.root_dir, self.feature_dir, f"val_{self.audio_feature}", self.training_datasets, self.eval_datasets, self.task,
                self.temporal_reduction, self.data_ratio, self.take_val, temporal_axis=self.temporal_axis,audio_feature=self.audio_feature)
            self.test_dataset = LPFeaturesDataset(self.root_dir, self.feature_dir, f"test_{self.audio_feature}", self.training_datasets, self.eval_datasets, self.task,
                self.temporal_reduction, 1.0, self.take_test, temporal_axis=self.temporal_axis,audio_feature=self.audio_feature)

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
