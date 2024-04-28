import os, random, argparse, math

def ff_split_dataset(directory, test_ratio=0.1, val_ratio=0.1, feat_type="MFCC"):
    """
    Split files in the /video directory into train, test, and validation sets.
    
    Args:
    - directory (str): Path to the directory containing the /video subdirectory.
    - test_ratio (float): Proportion of the dataset to include in the test split.
    - val_ratio (float): Proportion of the dataset to include in the validation split.
    """
    videos_dir = os.path.join(directory, 'cropped')
    files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
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
    write_filenames(train_files, os.path.join(directory, f'train_{feat_type}.txt'))
    write_filenames(test_files, os.path.join(directory, f'test_{feat_type}.txt'))
    write_filenames(val_files, os.path.join(directory, f'val_{feat_type}.txt'))


#NOTE: DeepfakeTIMIT has to be split such that there is no overlap of speakers among the train/eval/test splits
def split_DeepfakeTIMIT(root: str, test: float, val: float, feat_type:str):
    videos = list(filter(lambda x: x.endswith('.mp4') and os.path.exists(os.path.join(root, feat_type, x.replace(".mp4", ".npy"))),
                  os.listdir(os.path.join(root, 'cropped'))))
    speaker_files = {}
    # get video tracks per speaker id
    for filename in videos:
        speaker_id = filename.split('-')[0]
        if speaker_id not in speaker_files:
            speaker_files[speaker_id] = []
        speaker_files[speaker_id].append(filename)
    spk_id_list = list(speaker_files.keys())
    total_num = len(spk_id_list)
    train_ratio = 1-test-val
    val_ratio = train_ratio + val
    print("total num", total_num, int(total_num * train_ratio))
    # shuffle speakers
    random.shuffle(spk_id_list)
    with open(os.path.join(root, f"train_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * train_ratio)):
            for video in speaker_files[spk_id_list[i]]:
                f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"val_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * train_ratio), int(total_num * val_ratio)):
            for video in speaker_files[spk_id_list[i]]:
                f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"test_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * val_ratio), total_num):
            for video in speaker_files[spk_id_list[i]]:
                f.write(video[:-4] + "\n") 


import re

def get_first_id_number(filename):
    parts = filename.split("-")
    for part in parts:
        match = re.search(r'id(\d+)', part)
        if match:
            return match.group(1)
    return None


def split_DFDC(root: str, test: float, val: float, feat_type:str):
    videos = list(filter(lambda x: x.endswith('.mp4') and os.path.exists(os.path.join(root, feat_type, x.replace(".mp4", ".npy"))),
              os.listdir(os.path.join(root, 'cropped'))))
    f = filter(lambda x: x.endswith('.mp4'),os.listdir(os.path.join(root, 'test_video')))
    test_videos = list(f)
    
    train = []
    test = []

    for filename in videos:
        print(filename.split("-")[0] + ".mp4")
        if filename.split("-")[0] + ".mp4" in test_videos:
            test.append(filename)
    

    random.shuffle(videos)

    print(len(videos))

    split = len(videos) - int(math.ceil(len(videos) * val))
    train = videos[:split]
    val = videos[split:]

    random.shuffle(test)

    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    with open(os.path.join(root, f"train_{feat_type}.txt"), "w") as f:
        for video in train:
            f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"val_{feat_type}.txt"), "w") as f:
        for video in val:
            f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"test_{feat_type}.txt"), "w") as f:
        for video in test:
            f.write(video[:-4] + "\n")

def split_RAVDESS(root: str, test: float, val: float, feat_type:str):
    videos = list(filter(lambda x: x.endswith('.mp4') and os.path.exists(os.path.join(root, feat_type, x.replace(".mp4", ".npy"))),
              os.listdir(os.path.join(root, 'cropped'))))
    
    test_speakers = ["05", "10", "16", "19"]
    val_speakers = ["11", "13", "18", "22"]
    train = []
    test = []
    val = []
    for filename in videos:
        speaker_id = filename.split('-')[-2]
        if speaker_id in test_speakers:
            test.append(filename)
        elif speaker_id in val_speakers:
            val.append(filename)
        else:
            train.append(filename)


    random.shuffle(train)
    random.shuffle(test)
    random.shuffle(val)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    with open(os.path.join(root, f"train_{feat_type}.txt"), "w") as f:
        for video in train:
            f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"val_{feat_type}.txt"), "w") as f:
        for video in val:
            f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"test_{feat_type}.txt"), "w") as f:
        for video in test:
            f.write(video[:-4] + "\n")


#NOTE: FakeAVCeleb has to be split such that there is no overlap of speakers among the train/eval/test splits
def split_fakeAVCeleb(root: str, test: float, val: float, feat_type:str):
    videos = list(filter(lambda x: x.endswith('.mp4') and os.path.exists(os.path.join(root, feat_type, x.replace(".mp4", ".npy"))),
              os.listdir(os.path.join(root, 'cropped'))))
    speaker_files = {}
    # get video tracks per speaker id
    for filename in videos:
        speaker_id = "id"+get_first_id_number(filename) # retrieve speakerID from filename
        print("speaker_id", speaker_id) 
        if speaker_id not in speaker_files:
            speaker_files[speaker_id] = []
        speaker_files[speaker_id].append(filename)
    spk_id_list = list(speaker_files.keys())
    total_num = len(spk_id_list)
    train_ratio = 1-test-val
    val_ratio = train_ratio + val
    print("Total number of speakers", total_num, int(total_num * train_ratio))
    # shuffle speakers
    random.shuffle(spk_id_list)
    with open(os.path.join(root, f"train_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * train_ratio)):
            for video in speaker_files[spk_id_list[i]]:
                f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"val_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * train_ratio), int(total_num * val_ratio)):
            for video in speaker_files[spk_id_list[i]]:
                f.write(video[:-4] + "\n")

    with open(os.path.join(root, f"test_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * val_ratio), total_num):
            for video in speaker_files[spk_id_list[i]]:
                f.write(video[:-4] + "\n")
                

# General method to split a dataset purely on file count basis
def split_dataset(root: str, test: float, val: float, feat_type:str):
    videos = list(filter(lambda x: x.endswith('.mp4') and os.path.exists(os.path.join(root, feat_type, x.replace(".mp4", ".npy"))),
                  os.listdir(os.path.join(root, 'cropped'))))
    
    total_num = len(videos)
    train_ratio = 1-test-val
    val_ratio = train_ratio + val

    with open(os.path.join(root, f"train_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * train_ratio)):
            f.write(videos[i][:-4] + "\n")

    with open(os.path.join(root, f"val_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * train_ratio), int(total_num * val_ratio)):
            f.write(videos[i][:-4] + "\n")

    with open(os.path.join(root, f"test_{feat_type}.txt"), "w") as f:
        for i in range(int(total_num * val_ratio), total_num):
            f.write(videos[i][:-4] + "\n")

def delete_corrupted_files(files, filepath, audio_filepath):
    for split in files:
        with open(os.path.join(filepath, split), "r") as file:
            lines = file.readlines()

        filtered_lines = []
        for line in lines:
            filename = line.strip()
            audio_file = os.path.join(audio_filepath, f"{filename}.npy")

            if os.path.exists(audio_file):
                filtered_lines.append(line)

        with open(os.path.join(filepath, split), "w") as file:
            file.writelines(filtered_lines)
    




parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Root directory of Dataset to Process")
parser.add_argument("--test", type=float, default=.1)
parser.add_argument("--val", type=float, default=.1)
parser.add_argument("--feat_type", type=str, default="MFCC")
args = parser.parse_args()

assert args.test + args.val < 1, "test and val ratio too high"

if __name__ == '__main__':
    data_root = args.data_dir
    feat_type = args.feat_type


    if "Forensics++" in data_root:
        ff_split_dataset(data_root, args.test, args.val, feat_type)
    elif "DeepfakeTIMIT" in data_root:
        split_DeepfakeTIMIT(data_root, args.test, args.val, feat_type)
    elif "FakeAVCeleb" in data_root:
        split_fakeAVCeleb(data_root, args.test, args.val, feat_type)
    elif "RAVDESS" in data_root:
        split_RAVDESS(data_root, args.test, args.val, feat_type)
    elif "DFDC" in data_root:
        split_DFDC(data_root, args.test, args.val, feat_type)
    else:
        split_dataset(data_root, args.test, args.val, feat_type)
         
    assert os.path.exists(os.path.join(data_root, f"train_{feat_type}.txt")), "Something went wrong creating split files."
    files = [f"train_{feat_type}.txt", f"val_{feat_type}.txt", f"test_{feat_type}.txt"]
    delete_corrupted_files(files, data_root, os.path.join(data_root, feat_type))