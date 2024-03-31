

from eat_model.feature_extract.feature_extract import extract_features
import os
from tqdm import tqdm
import numpy as np

    
def extract_features_from_file(source_dir, target_dir, granularity="utterance", target_length=1024, checkpoint_dir="pretrained/audio/EAT_pretrained_AS2M.pt"):
    for file in tqdm(os.listdir(source_dir)):
        if file.endswith(".wav"):
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, file.replace(".wav", ".npy"))
            extract_features({
                "source_file": source_file,
                "target_file": target_file,
                "model_dir": "EAT",
                "checkpoint_dir": checkpoint_dir,
                "granularity": granularity,
                "target_length": target_length,
                "norm_mean": -4.268,
                "norm_std": 4.569
            })
    

    return target_dir


def test_extracting_audio_features():
    
    try:
        extract_features({
            "source_file": "EAT/feature_extract/test.wav",
            "target_file": "EAT/feature_extract/test.npy",
            "model_dir": "EAT",
            "checkpoint_dir": "pretrained/audio/EAT_pretrained_AS2M.pt",
            "granularity": 'frame',
            "target_length": 512,
            "norm_mean": -4.268,
            "norm_std": 4.569
        })

        extracted = np.load("EAT/feature_extract/test.npy")
        print(f"Success, extracted features shape: {extracted.shape}")
    except Exception as e:
        assert False, f"Failed to extract features: {e}"

test_extracting_audio_features()
