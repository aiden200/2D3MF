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

from preprocess.eat_extract_audio_features import eat_extract_inference
from preprocess.extract_features import efficientFace_video_loader, extract_audio, extract_audio_xvectors, extract_audio_resnet




def efficientface_extraction(video_path, video_model, device):
    video_data = efficientFace_video_loader(video_path, device=device)
    video_embeddings = video_model.forward_features(video_data)
    return video_embeddings.detach()

def load_marlin_model(model, path=None):
    if model == "marlin_vit_small_ytf":
        if path:
            video_model = Marlin.from_file("marlin_vit_small_ytf", path)
        else:
            video_model = Marlin.from_file("marlin_vit_small_ytf", "pretrained/marlin_vit_small_ytf.encoder.pt")
    elif model == "marlin_vit_base_ytf":
        if path:
            video_model = Marlin.from_file("marlin_vit_base_ytf", path)
        else:
            video_model = Marlin.from_file(
                "marlin_vit_base_ytf", "pretrained/marlin_vit_base_ytf.encoder.pt")
    elif model == "marlin_vit_large_ytf":
        if path:
            video_model = Marlin.from_file("marlin_vit_large_ytf", path)
        else:
            video_model = Marlin.from_file(
                "marlin_vit_large_ytf", "pretrained/marlin_vit_large_ytf.encoder.pt")
    else:
        raise ValueError("Wrong MARLIN model!")   
    
    return video_model

def load_audio_model(audio_model_name, path=None):
    if audio_model_name == "MFCC":
        audio_model = get_mfccs
    elif audio_model_name == "xvectors":
        if path:
            audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir=path)
        else:
            audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif audio_model_name == "resnet":
        audio_model = AudioResNet18()
        if path == None:
            audio_resnet_model_path = "pretrained/RAVDESS_bs_32_lr_0.001_ep_250_03-30-22-28-29.pth"
        else:
            audio_resnet_model_path = path
        audio_model.load_state_dict(torch.load(audio_resnet_model_path))
    elif audio_model_name == "emotion2vec":
        audio_model = Emotion2vec()
    elif audio_model_name == "eat":
        audio_model = None
    else:
        raise ValueError(f"Incorrect audio_backbone {audio_model}")
    
    return audio_model
        
        

def load_efficient_face_model(path=None, device="cpu"):
    if not path:
        path = "pretrained/EfficientFace_Trained_on_AffectNet7.pth.tar"
    assert os.path.exists(path), "Missing EfficientFace pretrained model!"
    video_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes=1, im_per_sample=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_feature_extractor(video_model, "pretrained/EfficientFace_Trained_on_AffectNet7.pth.tar", device=device)
    return video_model

def marlin_extraction(video_path, marlin_model, video_model):
    config = resolve_config(marlin_model)
    video_embeddings = video_model.extract_video(
            video_path, crop_face=False,
            sample_rate=config.tubelet_size, stride=config.n_frames,
            keep_seq=False)
    return video_embeddings


def forward_video_model(video_path, model_name, video_model, device="cpu"):
    if "marlin" in model_name:
        return marlin_extraction(video_path, model_name, video_model)
    else:
        return efficientface_extraction(video_path, video_model, device)


def forward_audio_model(audio_path, model_name, video_embeddings, audio_model):
    if model_name == "MFCC":
        audio_embeddings = extract_audio(audio_path, audio_model, video_embeddings.shape[0])
    elif model_name == "xvectors":
        audio_embeddings = extract_audio_xvectors(audio_path, audio_model, video_embeddings.shape[0])
    elif model_name == "resnet":
        audio_embeddings = extract_audio_resnet(audio_path, audio_model, video_embeddings.shape[0])
    elif model_name == "emotion2vec":
        audio_embeddings = extract_audio(audio_path, audio_model, video_embeddings.shape[0])
    elif model_name == "eat":
        if not os.path.exists("temp"):
            os.mkdir("temp")
        target_file = os.path.join("temp", "eat_features.npy")
        return_code = eat_extract_inference(audio_path, target_file)
        if return_code != 0:
            raise ValueError(f"Something went wrong in the eat audio extraction")
        audio_embeddings = np.load(target_file)
    else:
        raise ValueError(f"Incorrect audio_backbone {model_name}")
    
    return audio_embeddings
    



