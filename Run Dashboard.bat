@echo off
title Gift-Card Sentiment Dashboard
cd /d "%~dp0"

echo ============================================================
echo  Gift-Card Sentiment Dashboard
echo  Starting server, then opening your browser in a few seconds...
echo ============================================================
echo.

rem Open the browser after the server has had a moment to start.
start "" /b cmd /c "timeout /t 5 /nobreak >nul & start "" http://localhost:8501"

".venv\Scripts\streamlit.exe" run app.py --server.headless true

echo.
echo The dashboard server has stopped. You can close this window.
pause
