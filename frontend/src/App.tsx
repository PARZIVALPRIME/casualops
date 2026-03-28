import { useState, useRef, useEffect } from 'react';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import { processVoice, generateSummary } from './api/backend';
import type { SummaryResponse } from './api/backend';
import { RecordButton } from './components/RecordButton';
import { ProductCard } from './components/ProductCard';
import { ProcessGuide } from './components/ProcessGuide';
import { LanguageBadge } from './components/LanguageBadge';
import { AudioPlayer } from './components/AudioPlayer';
import { SummaryCard } from './components/SummaryCard';
import './styles/index.css';

interface Message {
  id: string;
  role: 'user' | 'ai';
  text_english: string;
  text_local: string;
}

interface Session {
  id: string;
  timestamp: number;
  messages: Message[];
  detectedLang: string;
  intent: string;
  audio: string;
  summary: SummaryResponse | null;
}

function App() {
  const { isRecording, liveTranscript, startRecording, stopRecording, analyser } = useAudioRecorder();
  
  const [sessions, setSessions] = useState<Session[]>([{
    id: Date.now().toString(),
    timestamp: Date.now(),
    messages: [],
    detectedLang: '',
    intent: '',
    audio: '',
    summary: null
  }]);
  
  const [currentSessionId, setCurrentSessionId] = useState<string>(sessions[0].id);
  
  const currentSession = sessions.find(s => s.id === currentSessionId)!;
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);

  const staffEndRef = useRef<HTMLDivElement>(null);
  const customerEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    staffEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    customerEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession.messages, isProcessing, liveTranscript]);

  const updateCurrentSession = (updates: Partial<Session>) => {
    setSessions(prev => prev.map(s => 
      s.id === currentSessionId ? { ...s, ...updates } : s
    ));
  };

  const handleNewSession = () => {
    const newSession: Session = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      messages: [],
      detectedLang: '',
      intent: '',
      audio: '',
      summary: null
    };
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
  };

  const handleRecordStart = async () => {
    try {
      await startRecording();
    } catch (e) {
      alert("Microphone access denied or error occurred.");
    }
  };

  const handleRecordStop = async () => {
    const audioBlob = await stopRecording();
    setIsProcessing(true);
    updateCurrentSession({ summary: null });

    try {
      const result = await processVoice(audioBlob);
      
      const userMsg: Message = {
        id: Date.now().toString() + '-user',
        role: 'user',
        text_english: result.english_translation || result.original_text,
        text_local: result.original_text
      };

      const aiMsg: Message = {
        id: Date.now().toString() + '-ai',
        role: 'ai',
        text_english: result.ai_response_english,
        text_local: result.ai_response_translated || result.ai_response_english
      };

      updateCurrentSession({
        messages: [...currentSession.messages, userMsg, aiMsg],
        detectedLang: result.detected_language,
        intent: result.detected_intent,
        audio: result.audio_base64
      });
      
    } catch (error) {
      console.error("Failed to process voice:", error);
      alert("Failed to process voice. Ensure backend is running.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSummarize = async () => {
    if (currentSession.messages.length === 0) return;
    
    setIsSummarizing(true);
    try {
      const convo = currentSession.messages.map(m => ({
        role: m.role,
        text: m.text_english
      }));
      const res = await generateSummary(convo, currentSession.detectedLang || 'en-IN');
      updateCurrentSession({ summary: res });
    } catch (e) {
      console.error(e);
      alert("Failed to generate summary.");
    } finally {
      setIsSummarizing(false);
    }
  };

  return (
    <div className="app-container">
      {/* LEFT SIDEBAR: Conversation History */}
      <div className="history-sidebar glass-panel">
        <div className="history-header">
          <h2>Sessions</h2>
          <button className="new-session-btn" onClick={handleNewSession}>+ New</button>
        </div>
        <div className="session-list">
          {sessions.map(session => (
            <div 
              key={session.id} 
              className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
              onClick={() => setCurrentSessionId(session.id)}
            >
              <div className="session-title">
                Session {new Date(session.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
              <div className="session-preview">
                {session.messages.length > 0 
                  ? session.messages[0].text_english.substring(0, 30) + '...'
                  : 'New interaction'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* MAIN CHAT AREA */}
      <div className="main-chat-area glass-panel">
        <div className="header">
          <h1>Frontline Desk <span>UBI</span></h1>
          <LanguageBadge language={currentSession.detectedLang} />
        </div>

        <div className="split-view-headers">
          <div className="split-header staff-header">STAFF VIEW (ENGLISH)</div>
          <div className="split-header customer-header">CUSTOMER VIEW ({currentSession.detectedLang?.toUpperCase() || 'LOCAL'})</div>
        </div>

        <div className="split-view-container">
          <div className="split-column staff-column">
            {currentSession.messages.map(msg => (
              <div key={`staff-${msg.id}`} className={`chat-bubble ${msg.role}`}>
                <div className="bubble-label">{msg.role === 'user' ? 'CUSTOMER' : 'AI'}</div>
                <div className="bubble-text">{msg.text_english}</div>
              </div>
            ))}
            {isProcessing && (
              <div className="loading-indicator">Processing voice input...</div>
            )}
            {isRecording && liveTranscript && (
              <div className="chat-bubble user" style={{ opacity: 0.7 }}>
                <div className="bubble-label">CUSTOMER (Live)</div>
                <div className="bubble-text">{liveTranscript}</div>
              </div>
            )}
            <div ref={staffEndRef} />
          </div>

          <div className="split-column customer-column">
            {currentSession.messages.map(msg => (
              <div key={`cust-${msg.id}`} className={`chat-bubble ${msg.role}`}>
                <div className="bubble-label">{msg.role === 'user' ? 'CUSTOMER' : 'AI'}</div>
                <div className="bubble-text">{msg.text_local}</div>
              </div>
            ))}
            {isProcessing && (
              <div className="loading-indicator">Processing...</div>
            )}
            {isRecording && liveTranscript && (
              <div className="chat-bubble user" style={{ opacity: 0.7 }}>
                <div className="bubble-label">CUSTOMER (Live)</div>
                <div className="bubble-text">{liveTranscript}</div>
              </div>
            )}
            <div ref={customerEndRef} />
          </div>
        </div>

        <div className="controls-section">
          <RecordButton 
            isRecording={isRecording}
            onStart={handleRecordStart}
            onStop={handleRecordStop}
            analyser={analyser}
          />
          
          <button 
            className="summarize-btn"
            onClick={handleSummarize}
            disabled={isRecording || isProcessing || isSummarizing || currentSession.messages.length === 0}
          >
            {isSummarizing ? "Summarizing..." : "End & Summarize"}
          </button>
        </div>
      </div>

      {/* SIDE PANEL */}
      <div className="side-panel">
        {currentSession.audio && (
          <div className="glass-panel">
            <h3>🔊 Response Audio</h3>
            <AudioPlayer base64Audio={currentSession.audio} autoPlay={true} />
          </div>
        )}

        {currentSession.intent && <ProductCard intent={currentSession.intent} />}
        {currentSession.intent && <ProcessGuide intent={currentSession.intent} />}
        {currentSession.summary && <SummaryCard summary={currentSession.summary} />}
        
        {!currentSession.intent && !currentSession.summary && !currentSession.audio && (
          <div className="glass-panel" style={{ opacity: 0.5, textAlign: 'center' }}>
            <p>Waiting for customer query...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;