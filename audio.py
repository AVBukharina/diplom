from pydub import AudioSegment
import os
import streamlit as st
from faster_whisper import WhisperModel

def to_mp3(audio, output_audio_file, download_path):
    audio_format = audio.name.split('.')[-1].lower()
    audio_data = AudioSegment.from_file(os.path.join(download_path, audio.name), audio_format)
    audio_data.export(os.path.join(download_path, output_audio_file), format="mp3")

    return output_audio_file

@st.cache_resource(show_spinner=True)
def load_models_tiny():
    with st.spinner(f"Загрузка модели Tiny"):
        model = WhisperModel('tiny', device="cpu", compute_type="int8")
    return model

@st.cache_resource(show_spinner=True)
def load_models_base():
    with st.spinner(f"Загрузка модели Base"):
        model = WhisperModel('base', device="cpu", compute_type="int8")
    return model

@st.cache_resource(show_spinner=True)
def load_models_small():
    with st.spinner(f"Загрузка модели Small"):
        model = WhisperModel('small', device="cpu", compute_type="int8")
    return model

@st.cache_resource(show_spinner=True)
def load_models_medium():
    with st.spinner(f"Загрузка модели Medium"):
        model = WhisperModel('medium', device="cpu", compute_type="int8")
    return model

@st.cache_resource(show_spinner=True)
def load_models_large():
    with st.spinner(f"Загрузка модели Large"):
        model = WhisperModel('large-v1', device="cpu", compute_type="int8")
    return model

def process_audio(filename, function_models):
    model = function_models
    segments, info = model.transcribe(filename, beam_size=5)
    result = ''
    for segment in segments:
        result += segment.text
    return result



