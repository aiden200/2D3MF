
import os
import subprocess
from tqdm import tqdm
import numpy as np
import argparse
from pydub import AudioSegment


    
def extract_features_from_file(source_dir, target_dir="data/eat_features", granularity="frame", target_length=1024, checkpoint_dir="pretrained/audio/EAT_pretrained_AS2M.pt"):
    os.makedirs("data", exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    print("Extracting audio features with eat")
    # print(target_length)

    processed_files = 0
    failed_files = 0

    files_to_process = []

    for file in os.listdir(source_dir):
        if file.endswith(".wav"):
            files_to_process.append(file)
        elif file.endswith(".mp3"):
            if file.replace(".mp3", ".wav") not in os.listdir(source_dir):
                files_to_process.append(file)

    
    for file in tqdm(files_to_process):
        if file.endswith(".wav") or file.endswith(".mp3"):
            if file.endswith(".mp3"):
                audio = AudioSegment.from_mp3(os.path.join(source_dir, file))
                file = file.replace(".mp3", ".wav")
                audio.export(os.path.join(source_dir, file), format="wav")
            
            source_file = os.path.join(source_dir, file)
            # parts = source_file.split(os.sep)
            # source_file = os.sep.join(parts[1:])

            stereo_audio = AudioSegment.from_wav(source_file)
            if stereo_audio.channels > 1:
                mono_audio = stereo_audio.set_channels(1)
                mono_audio.export(source_file, format="wav")

            if target_length == 1024:
                duration = 10000 # 10 seconds
            elif target_length == 512:
                duration = 5000 # 5 seconds
            else:
                raise ValueError("Wrong target length. Has to be 1024(10seconds) or 512(5) seconds.")
            
            audio = AudioSegment.from_file(source_file)

            if len(audio) > duration:
                # If the audio is longer than the target, trim it
                trimmed_audio = audio[:duration]
                trimmed_audio.export(source_file, format="wav")
            elif len(audio) < duration:
                # If the audio is shorter, calculate the needed padding
                silence_duration = duration - len(audio)
                # Create a segment of silence
                silence = AudioSegment.silent(duration=silence_duration)
                # Pad the audio with silence at the end
                padded_audio = audio + silence
                padded_audio.export(source_file, format="wav")
            
            audio = None

            target_file = os.path.join(target_dir, file.replace(".wav", ".npy"))

            # Construct the command to run the feature extraction script
            cmd = f"""
            cd src && python EAT/feature_extract/feature_extract.py \
                --source_file='../{source_file}' \
                --target_file='../{target_file}' \
                --model_dir='EAT' \
                --checkpoint_dir='../{checkpoint_dir}' \
                --granularity='{granularity}' \
                --target_length={target_length} \
                --norm_mean=-4.268 \
                --norm_std=4.569
            """
   

            # Execute the command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # print(target_file)
            # n = np.load(f"{target_file}")
            # print(n.shape)
            # Check if the command was executed successfully
            if result.returncode != 0 or not os.path.exists(target_file):
                print(f"Error processing {source_file}: {result.stderr}")
                failed_files+=1
            else:
                processed_files +=1
                # print(f"Processed {source_file} successfully.")
                # print(result.stdout)
    print(f"Processed: {processed_files} files.\nFailed to process: {failed_files} files.")
    
    return target_dir

def extract_features_eat(source_dir, target_dir, filename, granularity="frame", target_length=1024, checkpoint_dir="pretrained/audio/EAT_pretrained_AS2M.pt"):

    if filename.endswith(".mp3"):
        audio = AudioSegment.from_mp3(os.path.join(source_dir, filename))
        filename = filename.replace(".mp3", ".wav")
        audio.export(os.path.join(source_dir, filename), format="wav")
    
    source_file = os.path.join(source_dir, filename)
    stereo_audio = AudioSegment.from_wav(source_file)
    if stereo_audio.channels > 1:
        mono_audio = stereo_audio.set_channels(1)
        mono_audio.export(source_file, format="wav")

    if target_length == 1024:
        duration = 10000 # 10 seconds
    elif target_length == 512:
        duration = 5000 # 5 seconds
    else:
        raise ValueError("Wrong target length. Has to be 1024(10seconds) or 512(5) seconds.")
    
    audio = AudioSegment.from_file(source_file)

    if len(audio) > duration:
        trimmed_audio = audio[:duration]
        trimmed_audio.export(source_file, format="wav")
    elif len(audio) < duration:
        silence_duration = duration - len(audio)
        silence = AudioSegment.silent(duration=silence_duration)
        padded_audio = audio + silence
        padded_audio.export(source_file, format="wav")
    
    audio = None

    target_file = os.path.join(target_dir, filename.replace(".wav", ".npy"))

    # Construct the command to run the feature extraction script
    cmd = f"""
    cd src && python EAT/feature_extract/feature_extract.py \
        --source_file='../{source_file}' \
        --target_file='../{target_file}' \
        --model_dir='EAT' \
        --checkpoint_dir='../{checkpoint_dir}' \
        --granularity='{granularity}' \
        --target_length={target_length} \
        --norm_mean=-4.268 \
        --norm_std=4.569
    """


    # Execute the command
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(target_file):
        print(f"Error processing {source_file}: {result.stderr}")



def get_parser():
    parser = argparse.ArgumentParser(
        description="extract EAT features for downstream tasks"
    )
    parser.add_argument('--source_file', help='location of source wav files', default="src/EAT/feature_extract")
    parser.add_argument('--target_file', help='location of target npy files', default="data/eat_features")
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', default='pretrained/audio/EAT_pretrained_AS2M.pt')
    parser.add_argument('--target_length', type=int, help='the target length of Mel spectrogram in time dimension', default=1024)
    parser.add_argument('--norm_mean', type=float, help='mean value for normalization', default=-4.268)
    parser.add_argument('--norm_std', type=float, help='standard deviation for normalization', default=4.569)
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.source_file == "src/EAT/feature_extract":
        print("Test Extraction. Please specify directory in --target_file.")

    parts = args.source_file.split(os.sep)
    target_dir = os.sep.join(parts[:-1])

    extract_features_from_file(
        args.source_file, 
        target_dir=f"{target_dir}/eat_features",
        checkpoint_dir=args.checkpoint_dir, 
        target_length=args.target_length,
        )


if __name__ == '__main__':
    main()


