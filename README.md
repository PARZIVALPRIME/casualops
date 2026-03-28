# Frontline Banking Assistant Architecture

A production-ready, multilingual Gen-AI voice assistant crafted for Union Bank of India (UBI). This project delivers a comprehensive, intelligent, and highly secure voice banking experience, featuring a multi-agent LangGraph system with agentic RAG, consensus-based verification, and a robust 4-layer voice security shield.

## Key Features

- **Multilingual Gen-AI Voice Assistant:** Seamlessly understands and communicates in multiple Indian languages, breaking down language barriers for banking customers.
- **Multi-Agent LangGraph System:** Employs advanced LangGraph architecture to handle complex, multi-step banking queries by coordinating specialized AI agents.
- **Agentic RAG (Retrieval-Augmented Generation):** Fetches real-time, context-specific banking policies and user data to ground AI responses accurately.
- **Consensus-Based Verification:** Ensures extreme accuracy in transactions and critical information by requiring consensus among verifying AI agents before executing actions.
- **4-Layer Voice Security Shield:** Protects user data and prevents voice spoofing/deepfakes using state-of-the-art multi-factor voice authentication and liveness detection.
- **Premium React/TypeScript Frontend:** A sleek, dynamic UI featuring real-time audio visualization and contextual banking components, delivering a world-class user experience.

## Tech Stack

### Frontend
- **Framework:** React / Vite
- **Language:** TypeScript
- **Styling:** CSS3, Tailwind CSS (if applicable)
- **Features:** Real-time audio visualization, dynamic banking UI components.

### Backend
- **Language:** Python
- **Framework:** FastAPI
- **AI/Agents:** LangGraph, LangChain, OpenAI / specialized LLMs
- **Audio Processing:** Real-time speech-to-text (STT) and text-to-speech (TTS) integration.

## Project Structure

- `/frontend`: Contains the React/TypeScript frontend application.
- `/backend`: Contains the Python FastAPI backend, LangGraph agents, and integration logic.
- `start.bat`: A quick-start batch script for launching both frontend and backend development servers on Windows.

## Getting Started

### Prerequisites
- Node.js (for frontend)
- Python 3.10+ (for backend)

### Running Locally (Windows)

To start both the frontend and backend simultaneously, you can use the provided batch script:

```cmd
start.bat
```

This script will:
1. Start the frontend React development server at `http://localhost:3000`
2. Start the backend Python server at `http://localhost:8000` (Make sure your virtual environment `venv` is set up correctly in the parent directory or modifying the path accordingly).

### Manual Setup
**Backend Setup:**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

**Frontend Setup:**
```bash
cd frontend
npm install
npm run dev
```

## Security
This application is designed with bank-grade security considerations, utilizing a proprietary 4-Layer Voice Security Shield to defend against prompt injection, voice spoofing, and unauthorized access.
