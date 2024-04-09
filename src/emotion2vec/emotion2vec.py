from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import numpy as np
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
