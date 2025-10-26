@echo off
echo 🚀 Starting Optimizer Lens...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if pnpm is installed
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pnpm is not installed. Please install pnpm.
    exit /b 1
)

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Install Node dependencies
echo 📦 Installing Node dependencies...
pnpm install

REM Start the Python API in a new window
echo 🐍 Starting Python API on http://localhost:8000...
start "Optimizer Lens API" cmd /k "python core\api.py"

REM Wait for API to start
timeout /t 3 /nobreak >nul

REM Start the Next.js frontend
echo ⚛️  Starting Next.js frontend on http://localhost:3000...
pnpm dev
