from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import tempfile
import soundfile as sf


class Emotion2vec(object):
    def __init__(self, model_name="iic/emotion2vec_base", model_revision="v2.0.4"):
        self.inference_pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_name,
            model_revision=model_revision)

    def __call__(self, audio_input, sr=44100, granularity="utterance", extract_embedding=True):
        # Check if the input is a file path (str) or a NumPy array
        if isinstance(audio_input, list):
            # Change datatype from list to np array:
            audio_input = np.array(audio_input)
            # Handle NumPy array: save it as a temporary audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
                sf.write(tmpfile.name, audio_input, sr)
                audio_path = tmpfile.name
        else:
            # If it's a string, assume it's a file path
            audio_path = audio_input

        # Execute the pipeline with the provided (or temporary) audio file path
        rec_result = self.inference_pipeline(audio_path, granularity=granularity, extract_embedding=extract_embedding)

        # Optionally delete the temporary file if input was a NumPy array
        if isinstance(audio_input, list):
            os.remove(audio_path)
        return rec_result[0]['feats']


# Usage example:
if __name__ == '__main__':
    from preprocess.extract_features import *

    video_path = 'C:/Users/chun/Desktop/WorkSpace/2D3MF/preprocess/small_dataset/cropped/01-01-04-01-01-02-14.mp4'
    audio_path = 'C:/Users/chun/Desktop/WorkSpace/2D3MF/preprocess/small_dataset/audio/01-01-04-01-01-02-14.wav'
    video_model = Marlin.from_file("marlin_vit_base_ytf",
                                   "C:/Users/chun/Desktop/WorkSpace/2D3MF/preprocess/pretrained/marlin_vit_base_ytf.encoder.pt")
    config = resolve_config("marlin_vit_base_ytf")
    try:
        video_embeddings = video_model.extract_video(video_path,
                                                     crop_face=False,
                                                     sample_rate=config.tubelet_size,
                                                     stride=config.n_frames,
                                                     keep_seq=False)

        res = extract_audio(audio_path=audio_path,
                            audio_model=Emotion2vec(),
                            n_feats=video_embeddings.shape[0])

    except Exception as e:
        print(f"Video {video_path} error.", e)
