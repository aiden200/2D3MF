import os
import random
import argparse
import shutil


def yt_split_dataset(directory, test_ratio=0.1, val_ratio=0.1):
    """
    Split files in the /downloaded directory into train, test, and validation sets.
    
    Args:
    - directory (str): Path to the directory containing the /downloaded subdirectory.
    - test_ratio (float): Proportion of the dataset to include in the test split.
    - val_ratio (float): Proportion of the dataset to include in the validation split.
    """
    downloaded_dir = os.path.join(directory, 'downloaded')
    files = [f for f in os.listdir(downloaded_dir) if f.endswith('.mp4')]
    # random.shuffle(files)
    
    num_files = len(files)
    num_test = int(num_files * test_ratio)
    num_val = int(num_files * val_ratio)
    
    test_files = files[:num_test]
    val_files = files[num_test:num_test + num_val]
    train_files = files[num_test + num_val:]
    
    val_dp = test_files[:2]

    random.shuffle(test_files)
    random.shuffle(val_files)
    random.shuffle(train_files)

    assert(test_files[:2] != val_dp)


    # Function to write filenames to a file
    def write_filenames(filenames, file_path):
        with open(file_path, 'w') as file:
            for filename in filenames:
                file.write(f"{filename[:-4]}\n")  # Remove the '.mp4' extension
    
    # Write the splits to their respective files
    write_filenames(train_files, os.path.join(directory, 'train.txt'))
    write_filenames(test_files, os.path.join(directory, 'test.txt'))
    write_filenames(val_files, os.path.join(directory, 'val.txt'))


def combine_features(directory):
    """
    Combine all files from audio_features and audio_features_real into one directory.
    
    Args:
    - directory (str): Path to the directory containing the audio_features and audio_features_real subdirectories.
    """
    new_dir = "combined_features"

    audio_features_dir = os.path.join(directory, 'audio_features')
    audio_features_real_dir = os.path.join(directory, 'audio_features_real')
    combined_dir = os.path.join(directory, new_dir, 'audio_features')

    # Create the combined directory if it doesn't exist

    print(combined_dir)
    os.makedirs(combined_dir, exist_ok=True)

    # Copy files from audio_features to the combined directory
    for filename in os.listdir(audio_features_dir):
        if filename.endswith('.npy'):
            shutil.copy(os.path.join(audio_features_dir, filename), os.path.join(combined_dir, filename))

    # Copy files from audio_features_real to the combined directory
    for filename in os.listdir(audio_features_real_dir):
        if filename.endswith('.npy'):
            shutil.copy(os.path.join(audio_features_real_dir, filename), os.path.join(combined_dir, filename))

    video_features_dir = os.path.join(directory, 'marlin_vit_small_ytf')
    video_features_real_dir = os.path.join(directory, 'marlin_vit_small_ytf_real')
    combined_dir = os.path.join(directory, new_dir, 'marlin_vit_small_ytf')

    # Create the combined directory if it doesn't exist
    os.makedirs(combined_dir, exist_ok=True)

    # Copy files from video_features to the combined directory
    for filename in os.listdir(video_features_dir):
        if filename.endswith('.npy'):
            shutil.copy(os.path.join(video_features_dir, filename), os.path.join(combined_dir, f"{filename[:-4]}-1.npy"))

    # Copy files from video_features_real to the combined directory
    for filename in os.listdir(video_features_real_dir):
        if filename.endswith('.npy'):
            shutil.copy(os.path.join(video_features_real_dir, filename), os.path.join(combined_dir, f"{filename[:-4]}-0.npy"))

    adrian_ds_split_dataset(os.path.join(directory, new_dir))

def adrian_ds_split_dataset(directory, test_ratio=0.1, val_ratio=0.1):

    downloaded_dir = os.path.join(directory, 'marlin_vit_small_ytf')
    files = [f[:-4] for f in os.listdir(downloaded_dir) if f.endswith('.npy')]
    # random.shuffle(files)
    
    num_files = len(files)
    num_test = int(num_files * test_ratio)
    num_val = int(num_files * val_ratio)
    
    test_files = files[:num_test]
    val_files = files[num_test:num_test + num_val]
    train_files = files[num_test + num_val:]
    
    val_dp = test_files[:2]

    random.shuffle(test_files)
    random.shuffle(val_files)
    random.shuffle(train_files)

    assert(test_files[:2] != val_dp)


    # Function to write filenames to a file
    def write_filenames(filenames, file_path):
        with open(file_path, 'w') as file:
            for filename in filenames:
                file.write(f"{filename}\n")
    
    # Write the splits to their respective files
    write_filenames(train_files, os.path.join(directory, 'train.txt'))
    write_filenames(test_files, os.path.join(directory, 'test.txt'))
    write_filenames(val_files, os.path.join(directory, 'val.txt'))


if __name__ == "__main__":
    #python create_split.py --data_dir /path/to/data
    parser = argparse.ArgumentParser(description='Split dataset into train, test, and validation sets.')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt_mixed"),
                        help='Directory where the data is stored.')
    parser.add_argument('--data_type', type=str, default='yt',
                    help='Type of data processing')
    args = parser.parse_args()

    if args.data_type == 'yt':
        yt_split_dataset(args.data_dir)
    elif args.data_type == 'adrian_ds':
        combine_features(args.data_dir)
        # adrian_ds_split_dataset(args.data_dir)
    else:
        raise ValueError(f"Data type {args.data_type} not recognized.")