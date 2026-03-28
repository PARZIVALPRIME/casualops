const API_BASE = "http://localhost:8000/api";

export interface ProcessVoiceResponse {
  original_text: string;
  detected_language: string;
  english_translation: string;
  ai_response_english: string;
  ai_response_translated: string;
  audio_base64: string;
  detected_intent: string;
}

export interface SummaryResponse {
  summary_english: string;
  summary_translated: string;
}

export const processVoice = async (audioBlob: Blob): Promise<ProcessVoiceResponse> => {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  const response = await fetch(`${API_BASE}/process-voice`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
};

export const transcribeLive = async (audioBlob: Blob): Promise<string> => {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  try {
    const response = await fetch(`${API_BASE}/transcribe-live`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) return "";
    const data = await response.json();
    return data.transcript;
  } catch (e) {
    console.error(e);
    return "";
  }
};

export const generateSummary = async (
  conversation: { role: string; text: string }[],
  language: string
): Promise<SummaryResponse> => {
  const response = await fetch(`${API_BASE}/generate-summary`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ conversation, language }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
};