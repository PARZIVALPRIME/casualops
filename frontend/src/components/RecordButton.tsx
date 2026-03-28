import React, { useRef, useEffect } from 'react';

interface Props {
  isRecording: boolean;
  onStart: () => void;
  onStop: () => void;
  analyser: AnalyserNode | null;
}

export const RecordButton: React.FC<Props> = ({ isRecording, onStart, onStop, analyser }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    if (isRecording && analyser && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        animationRef.current = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#00ffcc';
        ctx.beginPath();

        const sliceWidth = (canvas.width * 1.0) / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * canvas.height) / 2;

          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);

          x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
      };

      draw();
    }

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isRecording, analyser]);

  return (
    <div className="record-container">
      {isRecording && (
        <canvas 
          ref={canvasRef} 
          width={200} 
          height={60} 
          className="waveform-canvas"
        />
      )}
      <button 
        className={`record-button ${isRecording ? 'recording' : ''}`}
        onClick={isRecording ? onStop : onStart}
      >
        {isRecording ? '🛑 STOP RECORDING' : '🎙️ RECORD'}
      </button>
    </div>
  );
};