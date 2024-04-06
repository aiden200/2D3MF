import os, random, argparse

def ff_split_dataset(directory, test_ratio=0.1, val_ratio=0.1):
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
    write_filenames(train_files, os.path.join(directory, 'train.txt'))
    write_filenames(test_files, os.path.join(directory, 'test.txt'))
    write_filenames(val_files, os.path.join(directory, 'val.txt'))

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




parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", help="Root directory of Dataset to Process")
parser.add_argument("--Forensics", action="store_true", help="Add flag when processing Forensics++")
parser.add_argument("--test", type=float, default=.1)
parser.add_argument("--val", type=float, default=.1)

args = parser.parse_args()


assert args.test + args.val < 1, "test and val ratio too high"

if __name__ == '__main__':
    data_root = args.data_dir

    if not os.path.exists(os.path.join(data_root, "train.txt")) or \
        not os.path.exists(os.path.join(data_root, "val.txt")) or \
        not os.path.exists(os.path.join(data_root, "test.txt")):
        if args.Forensics:
            ff_split_dataset(data_root, args.test, args.val)
        else:
            gen_split(data_root, args.test, args.val)
    
    assert os.path.exists(os.path.join(data_root, "train.txt")), "Something went wrong creating split files."