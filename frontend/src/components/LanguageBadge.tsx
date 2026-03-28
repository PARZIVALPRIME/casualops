import React from 'react';

interface Props {
  language: string;
}

export const LanguageBadge: React.FC<Props> = ({ language }) => {
  if (!language) return null;

  return (
    <div className="language-badge animate-fade-in">
      🌐 {language} Detected
    </div>
  );
};