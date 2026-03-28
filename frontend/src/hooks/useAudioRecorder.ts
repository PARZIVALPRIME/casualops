import { useState, useRef, useCallback } from 'react';
import { transcribeLive } from '../api/backend';

export const useAudioRecorder = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const startRecording = useCallback(async () => {
    try {
      setLiveTranscript('');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start(500);
      setIsRecording(true);

      timerRef.current = setInterval(async () => {
        if (audioChunksRef.current.length > 0) {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const text = await transcribeLive(blob);
          if (text) setLiveTranscript(text);
        }
      }, 1500);

      return analyser;
    } catch (err) {
      console.error("Error accessing microphone:", err);
      throw err;
    }
  }, []);

  const stopRecording = useCallback((): Promise<Blob> => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current) return;

      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
        
        if (audioContextRef.current) {
          audioContextRef.current.close();
        }

        setIsRecording(false);
        setLiveTranscript('');
        resolve(audioBlob);
      };

      mediaRecorderRef.current.stop();
    });
  }, []);

  return {
    isRecording,
    liveTranscript,
    startRecording,
    stopRecording,
    analyser: analyserRef.current
  };
};