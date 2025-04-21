import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from huggingface_hub import InferenceClient
import os

# Replace with your actual Hugging Face API key
API_KEY = "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"

def transcribe_microphone(api_key=API_KEY, model="openai/whisper-base"):
    """Transcribe audio from the microphone."""
    sample_rate = 16000
    duration = 10
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

def transcribe_file(file_path, api_key=API_KEY, model="openai/whisper-base"):
    """Transcribe audio from a .wav file."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    return transcribe_audio(file_path, api_key, model)

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

def save_transcription(text, filename="transcription.txt"):
    """Save transcribed text to a file."""
    if text:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcription saved to {filename}")
    else:
        print("No transcription to save.")

def main():
    """Main function to run the speech-to-text program."""
    print("Speech-to-Text Converter (Hugging Face API) 🎙️")
    choice = input("Enter 'mic' for microphone or 'file' for audio file: ").lower()
    if choice == "mic":
        text = transcribe_microphone()
    elif choice == "file":
        file_path = input("Enter path to .wav file (e.g., sample_audio.wav): ")
        text = transcribe_file(file_path)
    else:
        print("Invalid choice. Please enter 'mic' or 'file'.")
        return
    save_transcription(text)

if __name__ == "__main__":
    main()