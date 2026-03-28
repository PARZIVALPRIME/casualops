import React from 'react';

interface Props {
  role: 'user' | 'ai';
  text: string;
  translation?: string;
}

export const ChatMessage: React.FC<Props> = ({ role, text, translation }) => {
  return (
    <div className={`chat-message ${role}`}>
      <div className="message-role">
        {role === 'user' ? 'CUSTOMER' : 'AI RESPONSE'}
      </div>
      <div className="message-content">
        <p className="primary-text">{text}</p>
        {translation && (
          <p className="secondary-text">
            {role === 'user' ? `EN: "${translation}"` : `Translated: "${translation}"`}
          </p>
        )}
      </div>
    </div>
  );
};