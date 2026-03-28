@echo off
echo Starting Frontline Desk - UBI Gen-AI Voice Assistant...

start cmd /k "cd frontend && npm run dev"
start cmd /k "..\venv\Scripts\python backend\main.py"

echo Services starting in background.
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
