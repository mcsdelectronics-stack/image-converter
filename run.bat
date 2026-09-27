@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.10+ and tick "Add to PATH".
  pause
  exit /b 1
)
python -c "import PIL, numpy, cv2" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies for the first run...
  python -m pip install -r requirements.txt
)
python main.py %*
