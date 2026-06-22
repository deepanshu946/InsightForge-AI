import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

OPENAI_TRANSCRIBE_MODEL = os.getenv(
    "OPENAI_TRANSCRIBE_MODEL",
    "gpt-4o-mini-transcribe"
)

MAX_TRANSCRIPTION_WORKERS = int(
    os.getenv("MAX_TRANSCRIPTION_WORKERS", "5")
)

_client = OpenAI()


def transcribe_chunk_openai(chunk_path: str) -> str:

    # Convert WAV/audio chunk to a compressed MP3 before upload.
    # This reduces network transfer time while maintaining excellent
    # transcription quality for speech.
    compressed_path = f"{chunk_path}.mp3"

    audio = AudioSegment.from_file(chunk_path)
    audio.export(
        compressed_path,
        format="mp3",
        bitrate="64k"
    )

    max_retries = 3

    try:
        for attempt in range(max_retries):
            try:
                with open(compressed_path, "rb") as audio_file:
                    transcript = _client.audio.transcriptions.create(
                        model=OPENAI_TRANSCRIBE_MODEL,
                        file=audio_file,
                    )

                return transcript.text

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n❌ OpenAI transcription failed after {max_retries} attempts")
                    print(f"Error: {e}\n")
                    raise

                wait_time = 2 ** attempt
                print(
                    f"⚠️ OpenAI transcription attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
    finally:
        if os.path.exists(compressed_path):
            os.remove(compressed_path)


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcr ipts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

   



def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to OpenAI or Sarvam depending on language choice.
    - english  → OpenAI GPT-4o Mini Transcribe
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_openai(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:

    engine = "Sarvam AI" if language.lower() == "hinglish" else "GPT-4o Mini Transcribe"
    print(f"Using {engine} for transcription.")

    max_workers = min(MAX_TRANSCRIPTION_WORKERS, len(chunks))
    print(f"Using {max_workers} parallel transcription workers.")

    def process_chunk(args):
        index, chunk = args
        print(f"Transcribing chunk {index + 1}/{len(chunks)}...")
        return transcribe_chunk(chunk, language=language)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        texts = list(executor.map(process_chunk, enumerate(chunks)))

    print("Transcription complete.")

    return " ".join(texts).strip()