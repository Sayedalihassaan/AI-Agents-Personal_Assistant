import os
import logging
import tempfile
from openai import OpenAI
from backend.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def speech_to_text(audio_bytes: bytes = None) -> str:
    """
    Takes audio bytes and transcribes them to text using OpenAI's Whisper model.

    Args:
        audio_bytes (bytes): The audio data to transcribe.

    Returns:
        str: The transcribed text.
    """
    logger.info("🔊 Processing speech to text...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    os.remove(tmp_path)
    print(transcript)
    return transcript.text

def text_to_speech(text: str, voice: str = "coral", speech_file_path: str = "./response.mp3") -> str:    
    """
    Takes a string of text and generates an audio file with the text spoken in the provided voice.

    Args:
        text (str): The text to generate speech for.
        voice (str, optional): The voice to use for the speech. Defaults to "coral".
        speech_file_path (str, optional): The path to write the generated speech file to. Defaults to "./response.mp3".

    Returns:
        str: The path to the generated speech file.
    """
    logger.info("🔊 Generating speech...")
    with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text
    ) as response:
        response.stream_to_file(speech_file_path)
    return speech_file_path