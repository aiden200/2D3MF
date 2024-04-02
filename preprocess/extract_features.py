import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config

from eat_extract_audio_features import extract_features_eat

def delete_corrupted_files(filepath, corrupted_files):
    files = ["test.txt", "train.txt", "val.txt"]
    for split in files:
        with open(os.path.join(filepath, split), "r") as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if line.strip() not in corrupted_files]

        with open(os.path.join(filepath, split), "w") as file:
            file.writelines(filtered_lines)

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ Feature Extraction")
    parser.add_argument("--video_backbone", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--audio_backbone", type=str, default="default")
    parser.add_argument("--real_only", action='store_true')
    args = parser.parse_args()

    # model = Marlin.from_online(args.backbone)
    if args.backbone == "marlin_vit_small_ytf":
        model = Marlin.from_file("marlin_vit_small_ytf", "pretrained/marlin_vit_small_ytf.encoder.pt")
    elif args.backbone == "marlin_vit_base_ytf":
        model = Marlin.from_file("marlin_vit_base_ytf", "pretrained/marlin_vit_base_ytf.encoder.pt")
    elif args.backbone == "marlin_vit_large_ytf":
        model = Marlin.from_file("marlin_vit_large_ytf", "pretrained/marlin_vit_large_ytf.encoder.pt")
    else:
        raise ValueError(f"Incorrect backbone {args.backbone}")
    config = resolve_config(args.backbone)
    feat_dir = args.backbone

    model.cuda()
    model.eval()

    raw_video_path = os.path.join(args.data_dir, "cropped")
    raw_audio_path = os.path.join(args.data_dir, "audio")
    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))
    Path(os.path.join(args.data_dir, feat_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.data_dir, "audio_features")).mkdir(parents=True, exist_ok=True)

    corrupted_files = []

    for video_name in tqdm(all_videos):
        video_path = os.path.join(raw_video_path, video_name)
        audio_name = video_name.split("_")[0]
        audio_name = audio_name.split("-")[0]
        add_data_point = True
        if args.real_only:
            add_data_point = video_name.split("-")[-1][0] == "0"
        
        if add_data_point:
            audio_path = os.path.join(raw_audio_path, audio_name + ".mp3")
            save_path = os.path.join(args.data_dir, feat_dir, video_name.replace(".mp4", ".npy"))
            try:
                feat, audio_feat = model.extract_video_and_audio(
                    video_path, crop_face=False,
                    sample_rate=config.tubelet_size, stride=config.n_frames,
                    keep_seq=False, audio_path=audio_path)
                # save video features
                np.save(save_path, feat.cpu().numpy())
                # save audio features
                audio_save_path = os.path.join(args.data_dir, "audio_features_real_only", video_name.replace(".mp4", ".npy"))
                np.save(audio_save_path, audio_feat.cpu().numpy())

            except Exception as e:
                print(f"Video {video_path} error.", e)
                corrupted_files.append(video_name[:-4])
                #feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32)
                #audio_feat = torch.zeros(10, 87, dtype=torch.float32)
                continue
    
    delete_corrupted_files(args.data_dir, corrupted_files)
    print(f"Files Corrupted and ignored: {len(corrupted_files)}")
