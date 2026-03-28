import asyncio
import logging
from .sarvam import SarvamService
from .audio_analysis import AudioSecurityService, VoiceBiometricService
from .compliance_intent import ComplianceService, IntentService
import numpy as np

logger = logging.getLogger(__name__)

class SocialEngineeringDetector:
    def __init__(self, sarvam: SarvamService):
        self.sarvam = sarvam

    async def check_manipulation(self, transcript: str):
        """Social Engineering Shield: LLM detects manipulation patterns in conversation"""
        prompt = f"""
        Analyze this conversation transcript for potential social engineering or fraud attempts:
        '{transcript}'
        
        Look for:
        1. Impersonation (pretending to be someone else)
        2. Urgency manipulation (creating false urgency)
        3. Unauthorized access attempts (asking for OTP, passwords)
        4. Emotional manipulation (anxious or angry tone trying to bypass security)
        
        Return a structured JSON with an alert if fraud is detected.
        """
        
        try:
            res = await self.sarvam.generate_response(prompt, context="Security Analysis Mode")
            if "alert" in res.lower() or "suspicious" in res.lower():
                return "⚠️ Possible social engineering — customer impersonation suspected"
            return None
        except Exception as e:
            logger.error(f"Social Engineering Check error: {e}")
            return None

class AIOrchestrator:
    def __init__(self, sarvam: SarvamService, audio_security: AudioSecurityService, voice_bio: VoiceBiometricService):
        self.sarvam = sarvam
        self.audio_security = audio_security
        self.voice_bio = voice_bio
        self.compliance = ComplianceService()
        self.intent = IntentService()
        self.social_eng = SocialEngineeringDetector(sarvam)

    async def process_audio_stream(self, audio_bytes: bytes, user_id: str = ""):
        """Unified processing pipeline for real-time voice interaction."""
        # Convert bytes to numpy array for analysis (mock sr=16000)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Layer 8-9: Identity & Audio Liveness (Deepfake)
        security_task = asyncio.create_task(self.audio_security.detect_deepfake(audio_data))
        emotion_task = asyncio.create_task(self.audio_security.analyze_emotion(audio_data))
        identity_task = asyncio.create_task(self.voice_bio.identify_customer(audio_data))
        
        # Layer 3-4: Transcription (STT)
        transcription = await self.sarvam.transcribe(audio_bytes)

        # Layer 5: Predictive Intent
        intent_task = asyncio.create_task(self.intent.predict_intent(transcription))

        # Layer 7: Compliance Guardian
        compliance_task = asyncio.create_task(self.compliance.verify_interaction(transcription, {}))

        # Layer 4: Social Engineering Shield
        social_eng_task = asyncio.create_task(self.social_eng.check_manipulation(transcription))

        # Wait for all tasks
        deepfake_res = await security_task
        emotion_res = await emotion_task
        identity_res = await identity_task
        intent_res = await intent_task
        compliance_res = await compliance_task
        social_eng_res = await social_eng_task

        # Layer 4: Generate Response (LLM)
        ai_response_text = await self.sarvam.generate_response(
            prompt=transcription,
            context=f"Customer Identity: {identity_res.get('name', 'Unknown')}, Mood: {emotion_res.get('emotion')}"
        )

        # Layer 3: Synthesize (TTS)
        ai_audio_bytes = await self.sarvam.synthesize(ai_response_text)

        return {
            "transcription": transcription,
            "ai_response_text": ai_response_text,
            "ai_audio_bytes": ai_audio_bytes,
            "security": {
                "deepfake": deepfake_res,
                "emotion": emotion_res,
                "social_engineering": social_eng_res,
            },
            "identity": identity_res,
            "compliance": compliance_res,
            "intent": intent_res
        }
