import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import google.generativeai as genai
import os
import streamlit as st
from huggingface_hub import InferenceClient

# Your Hugging Face API Key for voice-to-text
HF_API_KEY = "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"

# Your Gemini API Key
GEMINI_API_KEY = "AIzaSyAgONi5KBlTLIS2McpU6YKu6lDU_3nOPSk"

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-exp-03-25")


# Function to transcribe audio from the microphone
def transcribe_microphone(api_key=HF_API_KEY, model="openai/whisper-base"):
    """Transcribe audio from the microphone."""
    sample_rate = 16000
    duration = 10  # Recording for 10 seconds
    print("Speak now! (Recording for 10 seconds)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    audio = audio.flatten()
    temp_wav = "temp_mic_audio.wav"
    wavfile.write(temp_wav, sample_rate, audio)
    text = transcribe_audio(temp_wav, api_key, model)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    return text


# Function to transcribe audio file using Hugging Face API
def transcribe_audio(file_path, api_key, model):
    """Transcribe audio using Hugging Face Inference API."""
    try:
        client = InferenceClient(api_key=api_key)
        print("Transcribing with Hugging Face API...")
        with open(file_path, "rb") as audio_file:
            audio_data = audio_file.read()
        result = client.automatic_speech_recognition(audio_data, model=model)
        text = result["text"]
        print("Transcription:", text)
        return text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


# Function to send message to Gemini and get a response
def get_gemini_response(user_input):
    """Send user input to Gemini API and get response."""
    chat = model.start_chat(history=[])
    response = chat.send_message(user_input)
    return response.text.strip()


# Streamlit UI to interact with the chatbot
st.title("Voice-based Chatbot with Gemini 2.5 Pro")
st.write("Speak into the microphone to chat with Gemini!")

if st.button("Start Microphone"):
    st.write("Recording your voice...")
    # Record audio and convert to text
    transcribed_text = transcribe_microphone()
    st.write(f"Transcribed Text: {transcribed_text}")
    
    if transcribed_text:
        # Send transcribed text to Gemini and get response
        gemini_response = get_gemini_response(transcribed_text)
        st.write(f"Chatbot Response: {gemini_response}")
    else:
        st.write("No text transcribed from microphone.")
