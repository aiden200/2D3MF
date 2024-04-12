from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import numpy as np
import tempfile
import soundfile as sf
import torch


class Emotion2vec(object):
    def __init__(self, model_name="iic/emotion2vec_base", model_revision="v2.0.4"):
        self.inference_pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_name,
            model_revision=model_revision)

    def __call__(self, audio_input, sr=44100, granularity="utterance", extract_embedding=True):
        # Check if the audio_input is empty
        if audio_input.size == 0:
            # Pad with zeros if the array is empty
            audio_input = np.zeros(sr, dtype=np.float32)

        audio_input = audio_input

        # Execute the pipeline with the provided (or temporary) audio file path
        rec_result = self.inference_pipeline(audio_input, granularity=granularity, extract_embedding=extract_embedding)

        return rec_result[0]['feats']
