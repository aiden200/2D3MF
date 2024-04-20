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
from audio_resnet.audio_resnet18 import AudioResNet18
from emotion2vec.emotion2vec import Emotion2vec


# Used to get speech xvector embeddings
from speechbrain.inference.speaker import EncoderClassifier

from eat_extract_audio_features import extract_features_eat

def efficientface_video_extraction(video_save_path, video_model, video_path):
    if os.path.exists(video_save_path): 
        video_embeddings = np.load(video_save_path)
    else:
        #TODO
        #video_embeddings = video_model.extract_video

        np.save(video_save_path, video_embeddings.cpu().numpy())
    
    return video_embeddings

def ff_check_real_audio_loaded(video_name, dataset_dir, feat_dir_audio):
    real_fake_token = video_name.split("-")[-1][0]
    if real_fake_token == "1":
        #checking for real video loaded
        prefix = video_name.split("_")[0]
        audio_save_path = os.path.join(dataset_dir, feat_dir_audio, f"{prefix}.npy")
        if os.path.exists(audio_save_path):
            audio = np.load(audio_save_path)
            audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
            np.save(audio_save_path, audio)
            return True
    return False

def marlin_video_extraction(save_path, video_model, video_path, config):
    if os.path.exists(save_path): # simply load MARLIN embedding
        video_embeddings = np.load(save_path)
    else: # else extract MARLIN embedding
        video_embeddings = video_model.extract_video(
            video_path, crop_face=False,
            sample_rate=config.tubelet_size, stride=config.n_frames,
            keep_seq=False)
        # save video embeddings
        np.save(save_path, video_embeddings.cpu().numpy())
    
    return video_embeddings

def get_eat(video_name, dataset_dir, raw_audio_path, video_path, audio_name):


    if not os.path.exists(os.path.join(dataset_dir, "eat", audio_name + ".npy")):
        print(os.path.join(dataset_dir, "eat", audio_name + ".npy"))
        audio_save_path = os.path.join(dataset_dir, "eat")
        return_code = extract_features_eat(raw_audio_path, audio_save_path, audio_name + ".wav", new_filename=video_name.replace(".mp4", ".npy"))
        if return_code != 0:
            print(f"Eat feature extraction: {video_path} error.")
    
            return video_name[:-4]
    return 0


def delete_corrupted_files(filepath, corrupted_files):
    files = ["test.txt", "train.txt", "val.txt"]
    for split in files:
        with open(os.path.join(filepath, split), "r") as file:
            lines = file.readlines()

        filtered_lines = [
            line for line in lines if line.strip() not in corrupted_files]

        with open(os.path.join(filepath, split), "w") as file:
            file.writelines(filtered_lines)

def delete_corrupted_files_in_folder(filepath, folder):
    files = ["test.txt", "train.txt", "val.txt"]
    for split in files:
        with open(os.path.join(filepath, split), "r") as file:
            lines = file.readlines()

        filtered_lines = [
            line for line in lines if f"{line.strip()}.npy" in os.listdir(folder)]
        # print(len(filtered_lines))
        with open(os.path.join(filepath, split), "w") as file:
            file.writelines(filtered_lines)

def check_dimensions_eat(filepath):
    folder = os.path.join(filepath, "eat_features")
    for f in os.listdir(folder):
        n = np.load(os.path.join(folder, f))
        assert n.shape == (512, 768)


def extract_audio_xvectors(audio_path, audio_model, n_feats):
    audio, sr = audio_load(audio_path)
    audio_features = []
    for i in range(n_feats):
        start_idx = int(i * sr)
        audio_buffer = audio[start_idx:start_idx + sr]  # take 1sec audio windows
        audio_feat = audio_model(torch.from_numpy(audio_buffer))  # compute embeddings, torch tensor needed
        audio_features.append(audio_feat[0]) # append xvector embeddings - shape (1, 7205)
    audio_features = [arr.unsqueeze(0) for arr in audio_features]
    audio_features = torch.cat(audio_features, dim=0)
    return audio_features  # (n_feats, n_embedding)


def extract_audio_resnet(audio_path, audio_model, n_feats):
    audio, sr = audio_load(audio_path)
    audio_features = []
    for i in range(n_feats):
        start_idx = int(i * sr)
        audio_buffer = audio[start_idx:start_idx + sr]  # take 1sec audio windows
        audio_feat = audio_model(torch.from_numpy(get_mfccs(audio_buffer)))  # compute embeddings using audio_model
        audio_features.append(audio_feat)
    audio_features = torch.cat(audio_features, dim=0)
    return audio_features.numpy(force=True)  # (n_feats, n_embedding)


def extract_audio(audio_path, audio_model, n_feats):
    audio, sr = audio_load(audio_path)
    audio_features = []
    for i in range(n_feats):
        start_idx = int(i * sr)
        audio_buffer = audio[start_idx:start_idx + sr]  # take 1sec audio windows
        audio_feat = audio_model(audio_buffer)  # compute embeddings using audio_model
        audio_features.append(audio_feat)
    audio_features = [torch.from_numpy(arr).unsqueeze(0) for arr in audio_features]
    audio_features = torch.cat(audio_features, dim=0)
    return audio_features  # (n_feats, n_embedding)

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dataset Feature Extraction")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--video_backbone", type=str)
    parser.add_argument("--audio_backbone", type=str, default="MFCC")
    parser.add_argument("--Forensics", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.data_dir
    
    assert os.path.exists(os.path.join(dataset_dir, "cropped")) and os.path.exists(os.path.join(dataset_dir, "audio")), "Missing dir cropped or audio"

    marlin_configurations = ["marlin_vit_small_ytf", "marlin_vit_base_ytf", "marlin_vit_large_ytf"]

    # VIDEO BACKBONE
    # model = Marlin.from_online(args.backbone)
    if args.video_backbone == "marlin_vit_small_ytf":
        video_model = Marlin.from_file(
            "marlin_vit_small_ytf", "pretrained/marlin_vit_small_ytf.encoder.pt")
    elif args.video_backbone == "marlin_vit_base_ytf":
        video_model = Marlin.from_file(
            "marlin_vit_base_ytf", "pretrained/marlin_vit_base_ytf.encoder.pt")
    elif args.video_backbone == "marlin_vit_large_ytf":
        video_model = Marlin.from_file(
            "marlin_vit_large_ytf", "pretrained/marlin_vit_large_ytf.encoder.pt")
    elif args.video_backbone == "efficientface":
        video_model = None #TODO
    else:
        raise ValueError(f"Incorrect backbone {args.video_backbone}")
    
    if args.video_backbone in marlin_configurations:
        config = resolve_config(args.video_backbone)
        raw_video_path = os.path.join(dataset_dir, "cropped")
    else:
        raw_video_path = os.path.join(dataset_dir, args.video_backbone)

    feat_dir_video = args.video_backbone

    if torch.cuda.is_available():
        video_model.cuda()
    video_model.eval()

    # AUDIO BACKBONE
    # Audio embedding extractors
    if args.audio_backbone == "MFCC":
        audio_model = get_mfccs
    elif args.audio_backbone == "xvectors":
        audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif args.audio_backbone == "resnet":
        audio_model = AudioResNet18()
        audio_resnet_model_path = "pretrained/RAVDESS_bs_32_lr_0.001_ep_250_03-30-22-28-29.pth"
        audio_model.load_state_dict(torch.load(audio_resnet_model_path))
        #TODO: Make sure to integrate this logic with the below resnet todo block
    elif args.audio_backbone == "emotion2vec":
        audio_model = Emotion2vec()
    elif args.audio_backbone == "eat":
        audio_model = get_eat
    else:
        raise ValueError(f"Incorrect audio_backbone {args.audio_backbone}")
    
    
    feat_dir_audio = args.audio_backbone
    raw_audio_path = os.path.join(dataset_dir, "audio")

    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))
    Path(os.path.join(dataset_dir, feat_dir_video)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dataset_dir, feat_dir_audio)).mkdir(parents=True, exist_ok=True)

    corrupted_files = []

    for video_name in tqdm(all_videos):
        video_save_path = os.path.join(dataset_dir, feat_dir_video, video_name.replace(".mp4", ".npy"))
        video_path = os.path.join(raw_video_path, video_name)
        alt_video_path = os.path.join(raw_video_path, f"{video_name.split('-')[0]}.mp4")

        audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
        audio_path = os.path.join(raw_audio_path, video_name.replace(".mp4", ".wav"))
        alt_audio_path = os.path.join(raw_audio_path, f"{video_name[:-4].split('-')[0]}.wav")
        if args.Forensics:
            audio_path = os.path.join(raw_audio_path, f"{video_name[:-4].split('_')[0]}.wav")            
        
        if not os.path.exists(audio_path) and os.path.exists(alt_audio_path):
            audio_path = alt_audio_path
        if not os.path.exists(video_path) and os.path.exists(alt_video_path):
            video_path = alt_video_path
        
        base_name = os.path.basename(audio_path)
        audio_name, _ = os.path.splitext(base_name)

        # Optionally, create .wav files if .mp3 files exists

        # Only extract video and audio if both exist
        if not all(os.path.exists(path) for path in [video_path, audio_path]):
            print(f"File {video_path} or {audio_path} does not exist!")
            continue 
        try:
            # Video Feature Extraction
            if args.video_backbone in marlin_configurations:
                video_embeddings = marlin_video_extraction(video_save_path, video_model, video_path, config)
            elif args.video_backbone == "efficientface":
                video_embeddings = efficientface_video_extraction(video_save_path, video_model, video_path)

        except Exception as e:
            print(f"Video {video_path} error.", e)
            corrupted_files.append(video_name[:-4])
            continue
        try:
            # Audio Feature Extraction
            dup = False
            if args.Forensics:
                # Check if the real audio is loaded
                dup = ff_check_real_audio_loaded(video_name, dataset_dir, feat_dir_audio)            
            if dup:
                continue
            
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

        except Exception as e:
            print(f"Audio {audio_path} error.", e)
            corrupted_files.append(video_name[:-4])
            continue
