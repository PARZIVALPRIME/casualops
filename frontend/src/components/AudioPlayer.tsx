import React, { useRef, useEffect } from 'react';

interface Props {
  base64Audio: string;
  autoPlay?: boolean;
}

export const AudioPlayer: React.FC<Props> = ({ base64Audio, autoPlay = true }) => {
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (autoPlay && audioRef.current && base64Audio) {
      audioRef.current.play().catch(e => console.error("Audio playback failed:", e));
    }
  }, [base64Audio, autoPlay]);

  if (!base64Audio) return null;

  const audioSrc = `data:audio/wav;base64,${base64Audio}`;

  return (
    <div className="audio-player">
      <audio ref={audioRef} controls src={audioSrc} />
    </div>
  );
};