import React from 'react';
import processesData from '../data/processes.json';

interface Props {
  intent: string;
}

export const ProcessGuide: React.FC<Props> = ({ intent }) => {
  const steps = (processesData as Record<string, string[]>)[intent];

  if (!steps || steps.length === 0) return null;

  return (
    <div className="process-guide glass-panel">
      <h3>🔄 Required Steps</h3>
      <div className="process-stepper">
        {steps.map((step, index) => (
          <React.Fragment key={index}>
            <div className={`process-step ${index === 0 ? 'completed' : ''}`}>
              <span className="step-label">Step {index + 1}:</span>
              <span className="step-text">{step}</span>
              {index === 0 && <span className="step-check">✓</span>}
            </div>
            {index < steps.length - 1 && <span className="step-arrow">→</span>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};