from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import json
import logging
from typing import List, Dict, Any

from config import SARVAM_API_KEY, GROQ_API_KEY
from services.sarvam import SarvamService
from services.groq_llm import GroqLLMService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Frontline Desk - UBI Voice Assistant")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
sarvam = SarvamService()
llm = GroqLLMService()

class Message(BaseModel):
    role: str
    text: str

class SummaryRequest(BaseModel):
    conversation: List[Message]
    language: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "Frontline Desk API"}

@app.post("/api/transcribe-live")
async def transcribe_live(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        stt_result = await sarvam.speech_to_text(audio_bytes)
        return {"transcript": stt_result.get("transcript", "")}
    except Exception as e:
        logger.error(f"Error in /api/transcribe-live: {e}")
        return {"transcript": ""}

@app.post("/api/process-voice")
async def process_voice(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        
        # 1. STT: audio -> text + language
        stt_result = await sarvam.speech_to_text(audio_bytes)
        original_text = stt_result.get("transcript", "")
        detected_language = stt_result.get("language_code", "en-IN")
        
        # Fallback empty check
        if not original_text.strip():
            return {
                "original_text": "",
                "detected_language": detected_language,
                "english_translation": "",
                "ai_response_english": "Could not hear properly.",
                "ai_response_translated": "I couldn't hear that. Please try again.",
                "audio_base64": "",
                "detected_intent": "general"
            }

        # 2. Translate to English (if not English)
        english_translation = original_text
        if "en" not in detected_language.lower():
            english_translation = await sarvam.translate(original_text, detected_language, "en-IN")

        # 3. LLM: generate response
        ai_full_response = await llm.generate_response(english_translation)
        
        # Extract intent from LLM response
        lines = ai_full_response.strip().split("\n")
        intent = "general"
        clean_response_lines = []
        for line in lines:
            if line.startswith("INTENT:"):
                intent = line.replace("INTENT:", "").strip()
            else:
                clean_response_lines.append(line)
                
        ai_response_english = "\n".join(clean_response_lines).strip()
        
        # 4. Translate response to customer's language
        ai_response_translated = ai_response_english
        if "en" not in detected_language.lower():
            ai_response_translated = await sarvam.translate(ai_response_english, "en-IN", detected_language)
            
        # 5. TTS: generate audio
        audio_base64 = await sarvam.text_to_speech(ai_response_translated, detected_language)

        return {
            "original_text": original_text,
            "detected_language": detected_language,
            "english_translation": english_translation,
            "ai_response_english": ai_response_english,
            "ai_response_translated": ai_response_translated,
            "audio_base64": audio_base64,
            "detected_intent": intent
        }
        
    except Exception as e:
        logger.error(f"Error in /api/process-voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-summary")
async def generate_summary(req: SummaryRequest):
    try:
        conv_list = [{"role": msg.role, "text": msg.text} for msg in req.conversation]
        summary_result = await llm.generate_summary(conv_list, req.language)
        return summary_result
    except Exception as e:
        logger.error(f"Error in /api/generate-summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
