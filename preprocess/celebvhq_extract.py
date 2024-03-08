import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config, read_yaml

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ Feature Extraction")
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--config", type=str)
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
    hp_config = read_yaml(args.config)
    temporal_axis = hp_config['temporal_axis']
    feat_dir = args.backbone

    model.cuda()
    model.eval()

    raw_video_path = os.path.join(args.data_dir, "cropped")
    raw_audio_path = os.path.join(args.data_dir, "audio")
    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))

    finished_audios = []

    Path(os.path.join(args.data_dir, feat_dir)).mkdir(parents=True, exist_ok=True)
    for video_name in tqdm(all_videos):
        video_path = os.path.join(raw_video_path, video_name)
        audio_path = os.path.join(raw_audio_path, video_path[:3] + ".mp3")
        save_path = os.path.join(args.data_dir, feat_dir, video_name.replace(".mp4", ".npy"))
        audio_save_path = os.path.join(args.data_dir, feat_dir, f"{video_path[:3]}.npy")
        if audio_save_path in finished_audios:
            audio_save_path = None
        finished_audios.append(audio_save_path)

        try:
            feat, audio_feat = model.extract_video_and_audio(
                video_path, crop_face=False,
                sample_rate=config.tubelet_size, stride=config.n_frames,
                keep_seq=False, reduction="none", audio_path=audio_path, temporal_axis=temporal_axis)

        except Exception as e:
            print(f"Video {video_path} error.", e)
            print(f"Audio {audio_path} error.", e)
            feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32)
        np.save(save_path, feat.cpu().numpy())
        if audio_feat:
            np.save(audio_save_path, audio_feat.cpu().numpy())
