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

# Used to get speech xvector embeddings
#from speechbrain.inference.speaker import EncoderClassifier

def delete_corrupted_files(filepath, corrupted_files):
    files = ["test.txt", "train.txt", "val.txt"]
    for split in files:
        with open(os.path.join(filepath, split), "r") as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if line.strip() not in corrupted_files]

        with open(os.path.join(filepath, split), "w") as file:
            file.writelines(filtered_lines)

def extract_audio(audio_path, audio_model, n_feats):
    audio, sr = audio_load(audio_path)
    audio_features = []
    for i in range(n_feats):
        start_idx = int(i * sr)
        audio_buffer = audio[start_idx:start_idx+sr] # take 1sec audio windows
        audio_feat = audio_model(audio_buffer) # compute embeddings using audio_model
        audio_features.append(audio_feat)
    audio_features = [torch.from_numpy(arr).unsqueeze(0) for arr in audio_features]
    audio_features = torch.cat(audio_features, dim=0)
    return audio_features # (n_feats, n_embedding)

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ Feature Extraction")
    parser.add_argument("--video_backbone", type=str)
    parser.add_argument("--audio_backbone", type=str, default="MFCC")
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    # Video embedding extractor (MARLIN)
    if args.video_backbone == "marlin_vit_small_ytf":
        video_model = Marlin.from_file("marlin_vit_small_ytf", "pretrained/marlin_vit_small_ytf.encoder.pt")
    elif args.video_backbone == "marlin_vit_base_ytf":
        video_model = Marlin.from_file("marlin_vit_base_ytf", "pretrained/marlin_vit_base_ytf.encoder.pt")
    elif args.video_backbone == "marlin_vit_large_ytf":
        video_model = Marlin.from_file("marlin_vit_large_ytf", "pretrained/marlin_vit_large_ytf.encoder.pt")
    else:
        raise ValueError(f"Incorrect video_backbone {args.video_backbone}")
    config = resolve_config(args.video_backbone)
    feat_dir_video = args.video_backbone

    video_model.cuda()
    video_model.eval()

    # Audio embedding extractors
    if args.audio_backbone == "MFCC":
        audio_model = get_mfccs
    elif args.audio_backbone == "xvectors":
        #audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        pass
    elif args.audio_backbone == "resnet":
        # TODO: Steve
        raise ValueError(f"Error: {args.audio_backbone} not yet implemented")
    elif args.audio_backbone == "emotion2vec":
        # TODO: Tom
        raise ValueError(f"Error: {args.audio_backbone} not yet implemented")
    else:
        raise ValueError(f"Incorrect audio_backbone {args.audio_backbone}")
    feat_dir_audio = args.audio_backbone
    
    raw_video_path = os.path.join(args.data_dir, "cropped")
    raw_audio_path = os.path.join(args.data_dir, "audio")
    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))
    Path(os.path.join(args.data_dir, feat_dir_video)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.data_dir, feat_dir_audio)).mkdir(parents=True, exist_ok=True)

    corrupted_files = []

    for video_name in tqdm(all_videos):
        video_path = os.path.join(raw_video_path, video_name)
        audio_path = os.path.join(raw_audio_path, video_name.replace(".mp4", ".wav"))
        save_path = os.path.join(args.data_dir, feat_dir_video, video_name.replace(".mp4", ".npy"))
        try:
            video_embeddings = video_model.extract_video(
                video_path, crop_face=False,
                sample_rate=config.tubelet_size, stride=config.n_frames,
                keep_seq=False)
            # save video embeddings
            np.save(save_path, video_embeddings.cpu().numpy())
            # save audio embeddings
            audio_embeddings = extract_audio(audio_path, audio_model, video_embeddings.shape[0])
            assert audio_embeddings.shape[0] == video_embeddings.shape[0], "Video and audio n_feats dimension do not match"
            audio_save_path = os.path.join(args.data_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
            np.save(audio_save_path, audio_embeddings.cpu().numpy())

        except Exception as e:
            print(f"Video {video_path} error.", e)
            corrupted_files.append(video_name[:-4])
            #feat = torch.zeros(0, video_model.encoder.embed_dim, dtype=torch.float32)
            #audio_feat = torch.zeros(10, 87, dtype=torch.float32)
            continue
    
    delete_corrupted_files(args.data_dir, corrupted_files)
    print(f"Files Corrupted and ignored: {len(corrupted_files)}")
