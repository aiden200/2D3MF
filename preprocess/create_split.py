import os
import random
import argparse

def split_dataset(directory, test_ratio=0.1, val_ratio=0.1):
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

if __name__ == "__main__":
    #python create_split.py --data_dir /path/to/data
    parser = argparse.ArgumentParser(description='Split dataset into train, test, and validation sets.')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt_mixed"),
                        help='Directory where the data is stored.')
    args = parser.parse_args()

    split_dataset(args.data_dir)