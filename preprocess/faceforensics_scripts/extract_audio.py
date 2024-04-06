import os
import json
import requests
from pytube import YouTube
import subprocess
from pathlib import Path
from os import listdir
from os.path import isfile, join
import argparse


def load_json_data(sub_dir, filename):
    """Loads JSON data from a file."""
    filepath = os.path.join(sub_dir, filename)
    with open(filepath, 'r') as file:
        return json.load(file)


def get_fps(video_path):
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-hide_banner"],
        text=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    lines = result.stderr.split('\n')
    for line in lines:
        if "fps" in line:
            fps_info = [x for x in line.split(',') if 'fps' in x]
            if fps_info:
                fps = fps_info[0].strip()
                return fps.split(' ')[0]
    return -1


def is_youtube_link_valid(url):
    """Checks if the YouTube link is valid. This is a placeholder function; you'll need to replace its logic."""
    try:
        response = requests.get(url)
        if "Video unavailable" in response.text:
            return False
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking URL {url}: {e}")
        return False


def download_audio_from_youtube(yt, output_path, filename):
    """Downloads audio from a YouTube video. Requires 'pytube'. Adjust according to your needs."""
    # yt = YouTube(url)
    # audio_stream = yt.streams.get_audio_only()
    # audio_stream.download(output_dir=output_path, filename=filename)

    audio_stream = yt.streams.get_audio_only()
    # Combine the output path and filename to form the full path
    full_path = os.path.join(output_path, filename)
    # Download the audio stream to the specified path
    audio_stream.download(output_path=full_path)


def extract_audio_segment(input_audio_path, output_audio_path, start_frame, end_frame, fps):
    """
    Extracts an audio segment from an audio file based on start and end frames.

    Args:
    input_audio_path (str): Path to the input audio file.
    output_audio_path (str): Path where the extracted audio segment will be saved.
    start_frame (int): The starting frame number.
    end_frame (int): The ending frame number.
    fps (float): Frames per second of the original video.
    """
    # Convert frames to seconds
    start_seconds = start_frame / fps
    end_seconds = end_frame / fps
    duration_seconds = end_seconds - start_seconds

    # Use ffmpeg to extract the audio segment
    command = [
        'ffmpeg',
        '-y', #overwritting file
        '-i', input_audio_path,  # Input file
        '-ss', str(start_seconds),  # Start time
        '-t', str(duration_seconds),  # Duration
        '-c:a', 'libmp3lame',  # Use the same codec
        output_audio_path  # Output file
    ]
    print(
        f'ffmpeg -y -i {command[3]} -ss {command[5]} -t {command[7]} -c:a libmp3lame {command[-1]}')
    
    subprocess.run(command, check=True)


def process_directory(curr_dir, video_dir, audio_dir):
    """Process each sub-directory within the given directory."""
    path = Path(audio_dir)
    unavailable = open(f"{path.parent.absolute()}/Unavailable_audios.txt", 'w')
    mappings = load_json_data('', 'conversion_dict.json')
    files = [f for f in listdir(video_dir) if isfile(join(video_dir, f))]

    for filename in files:
        print(f"Processing {filename}")
        fps = get_fps(f"{video_dir}/{filename}")
        filename = filename[:-4]  # mp4 deletion
        yt_convert = mappings.get(filename, "")
        print(yt_convert, fps)
        if not yt_convert or fps == -1:
            print(f"{filename} not working")
            continue
        yt_mapping_key = yt_convert[:-2]  # space 0 deletion
        yt_audio_dir = f"{audio_dir}/{yt_mapping_key}"
        data = load_json_data(yt_audio_dir, f"{yt_mapping_key}.json")
        youtube_url = data.get('webpage_url', '')
        yt = YouTube(youtube_url) 		
        try:
            yt.check_availability()
            if is_youtube_link_valid(youtube_url):
                print(f"URL is valid: {youtube_url}")
                frame_data = load_json_data(
                    f"{yt_audio_dir}", 'extracted_sequences/0.json')
                original_audio_name = f"original_{yt_mapping_key}"
                trimmed_audio_name = f"{filename}.mp3"
                download_audio_from_youtube(
                    yt, yt_audio_dir, original_audio_name)
                onlyfiles = [f for f in listdir(f"{yt_audio_dir}/{original_audio_name}") if isfile(
                    join(f"{yt_audio_dir}/{original_audio_name}", f))]
                input_file = f"{yt_audio_dir}/{original_audio_name}/{onlyfiles[0]}"
                extract_audio_segment(
                    input_file, f"{curr_dir}/audio_clips/{trimmed_audio_name}", frame_data[0], frame_data[-1], int(fps))
            else:
                unavailable.write(filename + " " + yt_mapping_key + f'\t{youtube_url}\n')
                print(
                    f"URL is not valid or video is not accessible: {youtube_url}")
        except Exception as e:
            unavailable.write(filename + " " + yt_mapping_key + f'\t{youtube_url}\n')
            print(
                f"URL is not valid or video is not accessible: {youtube_url}")
            
        
    unavailable.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some directories.')
    parser.add_argument('--dir', type=str, 
                        help='Directory containing video files')

    args = parser.parse_args()

    if not os.path.exists("audio_clips"):
        os.makedirs("audio_clips")
    
    video_dir = os.path.join(args.dir, "original_sequences/youtube/c23/videos")
    audio_dir = os.path.join(args.dir, "downloaded_videos_info")
    
    # curr_dir = os.path.dirname(os.path.realpath(__file__))
    process_directory(args.dir, video_dir, audio_dir)