import React from 'react';
import type { SummaryResponse } from '../api/backend';

interface Props {
  summary: SummaryResponse | null;
}

export const SummaryCard: React.FC<Props> = ({ summary }) => {
  if (!summary) return null;

  return (
    <div className="summary-card glass-panel animate-slide-up">
      <h3>📝 Interaction Summary</h3>
      <div className="summary-section">
        <p>{summary.summary_english}</p>
      </div>
    </div>
  );
};