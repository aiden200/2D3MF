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

sys.path.append(".")

if __name__ == '__main__':
    # delete_corrupted_files_in_folder("data/yt_av_mixed", "data/yt_av_mixed/eat_features")
    # check_dimensions_eat("data/yt_av_mixed")
    # exit(0)
    parser = argparse.ArgumentParser("CelebV-HQ Feature Extraction")
    parser.add_argument("--video_backbone", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--audio_backbone", type=str, default="default")
    parser.add_argument("--dataset", default="forensics++", type=str)
    parser.add_argument("--real_only", action='store_true')
    args = parser.parse_args()

    # model = Marlin.from_online(args.backbone)
    if args.video_backbone == "marlin_vit_small_ytf":
        model = Marlin.from_file(
            "marlin_vit_small_ytf", "pretrained/marlin_vit_small_ytf.encoder.pt")
    elif args.video_backbone == "marlin_vit_base_ytf":
        model = Marlin.from_file(
            "marlin_vit_base_ytf", "pretrained/marlin_vit_base_ytf.encoder.pt")
    elif args.video_backbone == "marlin_vit_large_ytf":
        model = Marlin.from_file(
            "marlin_vit_large_ytf", "pretrained/marlin_vit_large_ytf.encoder.pt")
    else:
        raise ValueError(f"Incorrect backbone {args.video_backbone}")
    

    if args.dataset == "forensics++":
        print(f"Feature extraction on forensics++")
    else:
        raise ValueError(f"Dataset extraction not implemented")

    config = resolve_config(args.video_backbone)
    feat_dir = args.video_backbone

    model.cuda()
    model.eval()

    raw_video_path = os.path.join(args.data_dir, "cropped")
    raw_audio_path = os.path.join(args.data_dir, "audio")

    all_videos = sorted(
        list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))
    
    Path(os.path.join(args.data_dir, feat_dir)).mkdir(
        parents=True, exist_ok=True)

    audio_save_dir = "audio_features"
    if args.audio_backbone == "eat":
        Path(os.path.join(args.data_dir, "eat_features")).mkdir(
            parents=True, exist_ok=True)
        audio_save_dir = "eat_features"

    Path(os.path.join(args.data_dir, audio_save_dir)).mkdir(
        parents=True, exist_ok=True)

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
            save_path = os.path.join(
                args.data_dir, feat_dir, video_name.replace(".mp4", ".npy"))
            try:
                if not os.path.exists(save_path):
                    feat, audio_feat = model.extract_video_and_audio(
                        video_path, crop_face=False,
                        sample_rate=config.tubelet_size, stride=config.n_frames,
                        keep_seq=False, audio_path=audio_path)
                    np.save(save_path, feat.cpu().numpy())


                dup = False
                
                if args.dataset == 'forensics++' and video_name.split("-")[-1][0] == "1":
                    #checking for real video loaded
                    prefix = video_name.split("_")[0]
                    audio_save_path = os.path.join(
                        args.data_dir, audio_save_dir, f"{prefix}.npy")
                    if os.path.exists(audio_save_path):
                        audio = np.load(audio_save_path)
                        audio_save_path = os.path.join(
                            args.data_dir, audio_save_dir, video_name.replace(".mp4", ".npy"))
                        np.save(audio_save_path, audio)
                        dup = True
                
                if dup:
                    continue

                if args.audio_backbone == "default":
                    audio_save_path = os.path.join(
                        args.data_dir, audio_save_dir, video_name.replace(".mp4", ".npy"))
                    np.save(audio_save_path, audio_feat.cpu().numpy())
                elif args.audio_backbone == "eat":
                    if not os.path.exists(os.path.join(
                            args.data_dir, audio_save_dir, audio_name + ".npy")):
                        return_code = extract_features_eat(raw_audio_path, os.path.join(
                            args.data_dir, audio_save_dir), audio_name + ".mp3", new_filename=video_name.replace(".mp4", ".npy"))
                        if return_code != 0:
                            print(f"Video {video_path} error.", e)
                            corrupted_files.append(video_name[:-4])

            except Exception as e:
                print(f"Video {video_path} error.", e)
                corrupted_files.append(video_name[:-4])
                # feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32)
                # audio_feat = torch.zeros(10, 87, dtype=torch.float32)
                continue

    delete_corrupted_files(args.data_dir, corrupted_files)
    print(f"Files Corrupted and ignored: {len(corrupted_files)}")
