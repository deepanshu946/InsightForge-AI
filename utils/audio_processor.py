import yt_dlp
import streamlit as st
#this will be used while chunking 
from pydub import AudioSegment
import os
import tempfile

def download_youtube_audio(url :str) ->str:
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    #it creates a .wav file in the download directory with the same name as the video title
     try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if info.get("requested_downloads"):
                downloaded = info["requested_downloads"][0]
                base_path = downloaded.get("filepath") or ydl.prepare_filename(info)
            else:
                base_path = ydl.prepare_filename(info)

            wav_path = os.path.splitext(base_path)[0] + ".wav"

        return wav_path

    except Exception as e:
        raise RuntimeError(
            f"Unable to download this YouTube video. It may be private, age-restricted, region-locked, or blocked by YouTube. Original error: {e}"
        )



#optimize the audio for better transcription accuracy by converting it to mono and resampling to 16kHz
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path


#convert to 10 minute chunks to avoid memory issues during transcription and to improve accuracy by providing smaller segments of audio for the model to process.
def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks



#trigger the entire processing pipeline based on the input source, whether it's a YouTube URL or a local file. It will return a list of paths to the processed audio chunks ready for transcription.
def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
        st.session_state.temp_wav_path = wav_path
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)

    try:
        if source.startswith("http://") or source.startswith("https://"):
            if os.path.exists(wav_path):
                os.remove(wav_path)
    except Exception as e:
        print(f"Warning: could not delete downloaded audio file: {e}")

    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
