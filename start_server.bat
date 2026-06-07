@echo off
echo ========================================
echo Blind Super-Resolution API Server
echo ========================================

echo.
echo Checking Python environment...
python --version

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting server on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.

python main.py
