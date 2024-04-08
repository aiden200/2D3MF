import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from audio_resnet.audio_resnet18 import AudioResNet18, get_audio_features
from eat_extract_audio_features import extract_features_eat
from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import get_mfccs, audio_load


# Used to get speech xvector embeddings
# from speechbrain.inference.speaker import EncoderClassifier


def ff_check_real_audio_loaded(video_name, dataset_dir, feat_dir_audio):
    real_fake_token = video_name.split("-")[-1][0]
    if real_fake_token == "1":
        # checking for real video loaded
        prefix = video_name.split("_")[0]
        audio_save_path = os.path.join(dataset_dir, feat_dir_audio, f"{prefix}.npy")
        if os.path.exists(audio_save_path):
            audio = np.load(audio_save_path)
            audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
            np.save(audio_save_path, audio)
            return True
    return False


def marlin_video_extraction(save_path, video_model, video_path, config):
    if os.path.exists(save_path):  # simply load MARLIN embedding
        # print("Loading pre-extracted marling embeding")
        video_embeddings = np.load(save_path)
        # print("shape", video_embeddings.shape)
    else:  # else extract MARLIN embedding
        video_embeddings = video_model.extract_video(
            video_path, crop_face=False,
            sample_rate=config.tubelet_size, stride=config.n_frames,
            keep_seq=False)
        # save video embeddings
        np.save(save_path, video_embeddings.cpu().numpy())

    return video_embeddings


def get_eat(video_name, dataset_dir, raw_audio_path, video_path):
    audio_name = video_name.split("_")[0]
    audio_name = audio_name.split("-")[0]
    if not os.path.exists(os.path.join(dataset_dir, "eat", audio_name + ".npy")):
        audio_save_path = os.path.join(dataset_dir, "eat")
        return_code = extract_features_eat(raw_audio_path, audio_save_path, audio_name + ".mp3",
                                           new_filename=video_name.replace(".mp4", ".npy"))
        if return_code != 0:
            print(f"Video {video_path} error.")

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
    # parser.add_argument("--dataset", default="Forensics", type=str)
    # parser.add_argument("--real_only", action='store_true')
    args = parser.parse_args()

    dataset_dir = args.data_dir

    assert os.path.exists(os.path.join(dataset_dir, "cropped")) and os.path.exists(
        os.path.join(dataset_dir, "audio")), "Missing dir cropped or audio"

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
    else:
        raise ValueError(f"Incorrect backbone {args.video_backbone}")

    config = resolve_config(args.video_backbone)
    feat_dir_video = args.video_backbone

    video_model.cuda()
    video_model.eval()

    # # DATASET SELECTION
    # root_dir = "2D3MF_Datasets"
    # if not os.path.exists(root_dir):
    #     os.mkdir(root_dir)

    # if args.dataset == "Forensics++":
    #     dataset_dir = os.path.join(root_dir, "Forensics++")
    # elif args.dataset == "DFDC":
    #     dataset_dir = os.path.join(root_dir, "DFDC")
    # elif args.dataset == "FakeAVCeleb":
    #     dataset_dir = os.path.join(root_dir, "FakeAVCeleb")
    # elif args.dataset == "DeepfakeTIMIT":
    #     dataset_dir = os.path.join(root_dir, "DeepfakeTIMIT")
    # else:
    #     raise ValueError(f"Dataset extraction not implemented please select one in (Forensics++, DFDC, FakeAVCeleb, DeepfakeTIMIT)")

    # if not os.path.exists(dataset_dir):
    #     os.mkdir(dataset_dir)
    #     print(f"Feature extraction on forensics++")

    # AUDIO BACKBONE
    # Audio embedding extractors
    if args.audio_backbone == "MFCC":
        audio_model = get_mfccs
    elif args.audio_backbone == "xvectors":
        # TODO: Implement
        pass
        # audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif args.audio_backbone == "resnet":
        audio_model = AudioResNet18()
        audio_resnet_model_path = "pretrained/RAVDESS_bs_32_lr_0.001_ep_250_03-30-22-28-29.pth"
        audio_model.load_state_dict(torch.load(audio_resnet_model_path))
    elif args.audio_backbone == "emotion2vec":
        # TODO: Tom
        raise ValueError(f"Error: {args.audio_backbone} not yet implemented")
    elif args.audio_backbone == "eat":
        audio_model = get_eat
    else:
        raise ValueError(f"Incorrect audio_backbone {args.audio_backbone}")

    feat_dir_audio = args.audio_backbone
    raw_video_path = os.path.join(dataset_dir, "cropped")
    raw_audio_path = os.path.join(dataset_dir, "audio")

    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))
    Path(os.path.join(dataset_dir, feat_dir_video)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dataset_dir, feat_dir_audio)).mkdir(parents=True, exist_ok=True)

    corrupted_files = []

    for video_name in tqdm(all_videos):
        video_path = os.path.join(raw_video_path, video_name)

        # TODO: not sure why real_only is needed
        add_data_point = True
        # if args.real_only:
        #    add_data_point = video_name.split("-")[-1][0] == "0"

        if add_data_point:
            audio_path = os.path.join(raw_audio_path, video_name.replace(".mp4", ".wav"))
            # Only extract video and audio if both exist
            if not all(os.path.exists(path) for path in [video_path, audio_path]):
                continue
            save_path = os.path.join(dataset_dir, feat_dir_video, video_name.replace(".mp4", ".npy"))
            try:
                # Video Feature Extraction
                video_embeddings = marlin_video_extraction(save_path, video_model, video_path, config)

                # Audio Feature Extraction
                dup = False
                if args.Forensics:
                    # Check if the real audio is loaded
                    dup = ff_check_real_audio_loaded(video_name, dataset_dir, feat_dir_audio)
                if dup:
                    continue

                if args.audio_backbone == "eat":
                    corrupted = audio_model(video_name, dataset_dir, raw_audio_path, video_path)
                    if corrupted != 0:
                        corrupted_files.append(corrupted)
                elif args.audio_backbone == "MFCC":  # For MFCC
                    # save audio embeddings
                    audio_embeddings = extract_audio(audio_path, audio_model, video_embeddings.shape[0])
                    assert audio_embeddings.shape[0] == video_embeddings.shape[
                        0], "Video and audio n_feats dimension do not match"
                    audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
                    np.save(audio_save_path, audio_embeddings)
                elif args.audio_backbone == "xvectors":
                    # TODO: Implement
                    pass
                elif args.audio_backbone == "resnet":
                    audio_features = get_audio_features(audio_path, 3)
                    audio_embeddings = audio_model(audio_features)
                    audio_save_path = os.path.join(dataset_dir, feat_dir_audio, video_name.replace(".mp4", ".npy"))
                    np.save(audio_save_path, audio_embeddings)
                elif args.audio_backbone == "emotion2vec":
                    # TODO: Implement
                    pass

            except Exception as e:
                print(f"Video {video_path} error.", e)
                corrupted_files.append(video_name[:-4])
                continue

    delete_corrupted_files(dataset_dir, corrupted_files)
    print(f"Files Corrupted and ignored: {len(corrupted_files)}")
