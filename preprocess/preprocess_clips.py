# parsing labels, segment and crop raw videos.
import argparse
import os
import sys
import random
import shutil
from faceforensics_scripts.create_split import ff_split_dataset

sys.path.append(os.getcwd())


def crop_face(root: str):
    from util.face_sdk.face_crop import process_videos
    source_dir = os.path.join(root, "downloaded")
    target_dir = os.path.join(root, "cropped")
    process_videos(source_dir, target_dir, ext="mp4")


def gen_split(root: str, test: float, val: float):
    videos = list(filter(lambda x: x.endswith('.mp4'), os.listdir(os.path.join(root, 'cropped'))))
    total_num = len(videos)
    train_ratio = 1-test-val
    val_ratio = train_ratio + val


    with open(os.path.join(root, "train.txt"), "w") as f:
        for i in range(int(total_num * train_ratio)):
            f.write(videos[i][:-4] + "\n")

    with open(os.path.join(root, "val.txt"), "w") as f:
        for i in range(int(total_num * train_ratio), int(total_num * val_ratio)):
            f.write(videos[i][:-4] + "\n")

    with open(os.path.join(root, "test.txt"), "w") as f:
        for i in range(int(total_num * val_ratio), total_num):
            f.write(videos[i][:-4] + "\n")

def find_file_by_prefix(directory, prefix):
    """
    Finds the first file in the specified directory that starts with the given prefix.
    
    Args:
    - directory (str): The path to the directory.
    - prefix (str): The prefix to search for.
    
    Returns:
    - str: The path to the first file found with the specified prefix, or None if no such file is found.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                return file
    return None


def reset_directory(directory):
    """
    Clears all contents of a specified directory without removing the directory itself.

    Args:
    - directory (str): The path to the directory to reset.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            # If item is a directory, remove it and all its contents
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            # If item is a file, remove it
            else:
                os.remove(item_path)
        except Exception as e:
            print(f"Error while deleting {item_path}: {e}")

def process_youtube_faces(root: str, mixed: bool = False)->str:
    original_sequences_root = f"{root}/original_sequences/youtube/c23/videos/"
    fake_sequences_root = f"{root}/manipulated_sequences/"
    new_sequences = f"Forensics++/"
    
    # if not os.path.exists(new_sequences):
    #     os.mkdir(new_sequences)
    #     os.mkdir(os.path.join(new_sequences, "downloaded/"))
    # else:
    #     reset_directory(f'{new_sequences}/downloaded/')
        
    
    filenames = []
    for root, dirs, files in os.walk(original_sequences_root):
        for file in files:
            if file[-4:] == ".mp4":
                filenames.append(file)
    print(f"Processing {len(filenames)} videos")
    
    deepfake_techniques = []
    ignore_techniques = ["DeepFakeDetection"]
    for root, dirs, files in os.walk(fake_sequences_root):
        for dir in dirs:
            if dir not in ignore_techniques:
                deepfake_techniques.append(dir)
        break
    print(deepfake_techniques)
            
    random.seed(22)
    random.shuffle(filenames) #mixing up the original videos
    
    real_videos = filenames[:len(filenames)//2]
    fake_videos = filenames[len(filenames)//2:]
    if mixed:
        real_videos = filenames
        fake_videos = filenames        
    
    vid_count = 0
    for vid_name in real_videos:
        src = f"{original_sequences_root}{vid_name}"
        dst = f"{new_sequences}downloaded/{vid_name[:-4]}-0.mp4"
        vid_count += 1
        shutil.copyfile(src, dst)
    
    print(f"Real videos: {vid_count}")

    vid_count = 0
    
    miss_count = 0
    for vid_name in fake_videos:
        technique = random.choice(deepfake_techniques)
        vid_name = find_file_by_prefix(f"{fake_sequences_root}{technique}/c23/videos/", vid_name[:-4])
        if vid_name:
            src = f"{fake_sequences_root}{technique}/c23/videos/{vid_name}"
            dst = f"{new_sequences}downloaded/{vid_name[:-4]}-1.mp4"
            vid_count += 1
            shutil.copyfile(src, dst)
        else:
            miss_count += 1
    print(f"Missed {miss_count} videos")
    print(f"Fake videos: {vid_count}")
    
    return new_sequences 


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Root directory of Dataset to Process")
parser.add_argument("--Forensics", action="store_true", help="Add flag when processing Forensics++")
parser.add_argument("--test", type=float, default=.1)
parser.add_argument("--val", type=float, default=.1)
args = parser.parse_args()


assert args.test + args.val < 1, "test and val ratio too high"


if __name__ == '__main__':
    data_root = args.data_dir
    crop_face(data_root)

    if not os.path.exists(os.path.join(data_root, "train.txt")) or \
        not os.path.exists(os.path.join(data_root, "val.txt")) or \
        not os.path.exists(os.path.join(data_root, "test.txt")):
        if args.Forensics:
            ff_split_dataset(data_root, args.test, args.val)
        else:
            gen_split(data_root, args.test, args.val)
    
    assert os.path.exists(os.path.join(data_root, "train.txt")), "Something went wrong creating split files."
