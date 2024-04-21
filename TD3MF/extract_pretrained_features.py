import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import get_mfccs, audio_load
from efficientFace.load_model import EfficientFaceTemporal, init_feature_extractor
from audio_resnet.audio_resnet18 import AudioResNet18
from emotion2vec.emotion2vec import Emotion2vec
from PIL import Image
import cv2


# Used to get speech xvector embeddings
from speechbrain.inference.speaker import EncoderClassifier

from preprocess.eat_extract_audio_features import extract_features_eat
from preprocess.extract_features import efficientFace_video_loader, get_eat




def efficientface_extraction(video_path):
    assert os.path.exists("pretrained/EfficientFace_Trained_on_AffectNet7.pth.tar"), "Missing EfficientFace pretrained model!"
    video_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes=1, im_per_sample=10)
    init_feature_extractor(video_model, "pretrained/EfficientFace_Trained_on_AffectNet7.pth.tar")
    video_data = efficientFace_video_loader(video_path)
    video_embeddings = video_model.forward_features(video_data)
    return video_embeddings.detach()

def marlin_extraction(video_path, marlin_model):
    if marlin_model == "marlin_vit_small_ytf":
        video_model = Marlin.from_file(
            "marlin_vit_small_ytf", "pretrained/marlin_vit_small_ytf.encoder.pt")
    elif marlin_model == "marlin_vit_base_ytf":
        video_model = Marlin.from_file(
            "marlin_vit_base_ytf", "pretrained/marlin_vit_base_ytf.encoder.pt")
    elif marlin_model == "marlin_vit_large_ytf":
        video_model = Marlin.from_file(
            "marlin_vit_large_ytf", "pretrained/marlin_vit_large_ytf.encoder.pt")
    else:
        raise ValueError("Wrong MARLIN model!")
    config = resolve_config(marlin_model)
    video_embeddings = video_model.extract_video(
            video_path, crop_face=False,
            sample_rate=config.tubelet_size, stride=config.n_frames,
            keep_seq=False)
    return video_embeddings

def forward_video_model(video_path, model):
    if "marlin" in model:
        return marlin_extraction(video_path, model)
    else:
        return efficientface_extraction(video_path)


def forward_audio_model(audio_path, model):
    if model == "MFCC":
        audio_model = get_mfccs
    elif model == "xvectors":
        audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif model == "resnet":
        audio_model = AudioResNet18()
        audio_resnet_model_path = "pretrained/RAVDESS_bs_32_lr_0.001_ep_250_03-30-22-28-29.pth"
        audio_model.load_state_dict(torch.load(audio_resnet_model_path))
    elif model == "emotion2vec":
        audio_model = Emotion2vec()
    elif model == "eat":
        audio_model = get_eat
    else:
        raise ValueError(f"Incorrect audio_backbone {model}")
    
    if args.audio_backbone == "eat":
        corrupted = audio_model(video_name, dataset_dir, raw_audio_path, video_path, video_name[:-4])
        if corrupted != 0:
            corrupted_files.append(corrupted)
    elif args.audio_backbone == "MFCC": # For MFCC
        # save audio embeddings
        audio_embeddings = extract_audio(audio_path, audio_model, video_embeddings.shape[0])
        assert audio_embeddings.shape[0] == video_embeddings.shape[0], "Video and audio n_feats dimension do not match"
        audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
        np.save(audio_save_path, audio_embeddings)
    elif args.audio_backbone == "xvectors":
        audio_embeddings = extract_audio_xvectors(audio_path, audio_model, video_embeddings.shape[0])
        assert audio_embeddings.shape[0] == video_embeddings.shape[0], "Video and audio n_feats dimension do not match"
        audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
        np.save(audio_save_path, audio_embeddings)
    elif args.audio_backbone == "resnet":
        audio_embeddings = extract_audio_resnet(audio_path, audio_model, video_embeddings.shape[0])
        assert audio_embeddings.shape[0] == video_embeddings.shape[0], "Video and audio n_feats dimension do not match"
        audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
        np.save(audio_save_path, audio_embeddings)
    elif args.audio_backbone == "emotion2vec":
            audio_embeddings = extract_audio(audio_path, audio_model, video_embeddings.shape[0])
            assert audio_embeddings.shape[0] == video_embeddings.shape[0], "Video and audio n_feats dimension do not match"
            audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
            np.save(audio_save_path, audio_embeddings)
