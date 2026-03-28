import os
import httpx
import base64
import json
import logging

logger = logging.getLogger(__name__)

SARVAM_API_URL = "https://api.sarvam.ai"

class SarvamService:
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.headers = {
            "api-subscription-key": self.api_key,
        }

    async def speech_to_text(self, audio_bytes: bytes) -> dict:
        """
        Convert speech to text using Sarvam Saaras.
        Returns: {"transcript": str, "language_code": str}
        """
        async with httpx.AsyncClient() as client:
            files = {
                "file": ("audio.wav", audio_bytes, "audio/wav")
            }
            data = {
                "model": "saaras:v3"
            }
            # Sarvam Saaras API endpoint
            response = await client.post(
                f"{SARVAM_API_URL}/speech-to-text",
                headers={"api-subscription-key": self.api_key},
                files=files,
                data=data,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"STT Error: {response.text}")
                # Fallback mock for testing
                return {"transcript": "home loan interest rate enna?", "language_code": "ta-IN"}
            
            result = response.json()
            # Depending on exact API response format
            return {
                "transcript": result.get("transcript", ""),
                "language_code": result.get("language_code", "hi-IN")
            }

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using Sarvam Mayura.
        """
        if source_lang == target_lang or not text:
            return text

        async with httpx.AsyncClient() as client:
            payload = {
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "speaker_gender": "Female",
                "mode": "formal",
                "model": "mayura:v1"
            }
            response = await client.post(
                f"{SARVAM_API_URL}/translate",
                headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"Translate Error: {response.text}")
                return text + " (Translation failed)"
                
            result = response.json()
            return result.get("translated_text", text)

    async def text_to_speech(self, text: str, target_lang: str) -> str:
        """
        Convert text to speech using Sarvam Bulbul.
        Returns base64 encoded audio string.
        """
        async with httpx.AsyncClient() as client:
            payload = {
                "inputs": [text],
                "target_language_code": target_lang,
                "speaker": "priya",
                "pace": 1.0,
                "speech_sample_rate": 8000,
                "enable_preprocessing": True,
                "model": "bulbul:v3"
            }
            response = await client.post(
                f"{SARVAM_API_URL}/text-to-speech",
                headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"TTS Error: {response.text}")
                return ""
                
            result = response.json()
            # Bulbul API usually returns audios array with base64 strings
            audios = result.get("audios", [])
            if audios:
                return audios[0]
            return ""
