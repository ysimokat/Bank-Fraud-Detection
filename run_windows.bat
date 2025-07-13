@echo off
REM Quick Start Script for Bank Fraud Detection System (Windows)
REM Author: Yanhong Simokat

echo ===========================================================
echo    Bank Fraud Detection System - Quick Start (Windows)
echo ===========================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [OK] Python detected
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check for dataset
if not exist "creditcard.csv" (
    echo.
    echo WARNING: Dataset not found!
    echo Please download creditcard.csv from:
    echo https://www.kaggle.com/mlg-ulb/creditcardfraud
    echo and place it in the project root directory.
    echo.
    pause
    exit /b 1
)

echo [OK] Dataset found
echo.

:menu
echo What would you like to run?
echo.
echo 1. Integrated Pipeline - ALL Models (Recommended)
echo 2. Quick Integrated Pipeline (Faster)
echo 3. Basic Fraud Detection Models Only
echo 4. Enhanced Deep Learning Models Only
echo 5. Professional Dashboard (Interactive)
echo 6. API Server
echo 7. GPU Configuration Test
echo 8. Jupyter Notebook Tutorials
echo 9. Install/Update Requirements
echo 0. Exit
echo.

set /p choice=Enter your choice (0-9): 

if "%choice%"=="1" (
    echo.
    echo Running Integrated Pipeline with ALL Models...
    echo This includes: Basic ML, XGBoost, LightGBM, Deep Learning, GNN
    python integrated_fraud_pipeline.py
    pause
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Running Quick Integrated Pipeline (Basic + Enhanced only)...
    python integrated_fraud_pipeline.py --quick
    pause
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo Running Basic Fraud Detection Models Only...
    python fraud_detection_models.py
    pause
    goto menu
)

if "%choice%"=="4" (
    echo.
    echo Running Enhanced Deep Learning Models Only...
    python enhanced_deep_learning.py
    pause
    goto menu
)

if "%choice%"=="5" (
    echo.
    echo Starting Professional Dashboard...
    echo Open http://localhost:8501 in your browser
    start http://localhost:8501
    python professional_fraud_dashboard.py
    pause
    goto menu
)

if "%choice%"=="6" (
    echo.
    echo Starting API Server...
    echo API will be available at http://localhost:8000
    echo Documentation at http://localhost:8000/docs
    start http://localhost:8000/docs
    python enhanced_fraud_api.py
    pause
    goto menu
)

if "%choice%"=="7" (
    echo.
    echo Testing GPU Configuration...
    python gpu_config.py
    pause
    goto menu
)

if "%choice%"=="8" (
    echo.
    echo Starting Jupyter Notebook...
    cd tutorials
    start http://localhost:8888
    jupyter notebook
    cd ..
    pause
    goto menu
)

if "%choice%"=="9" (
    echo.
    echo Installing/Updating Requirements...
    pip install -r requirements.txt
    pause
    goto menu
)

if "%choice%"=="0" (
    echo.
    echo Goodbye!
    pause
    exit /b 0
)

echo.
echo Invalid choice. Please try again.
echo.
pause
goto menu