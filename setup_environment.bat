@echo off
setlocal

:: --- Configuration ---
set VENV_NAME=.venv
set REQUIRED_PYTHON_VERSION=3.11
set SPIT_REPO_URL=https://github.com/dsb-ifi/SPiT.git
set YOLOV12_REPO_URL=https://github.com/sunsmarterjie/yolov12.git

echo ============================================
echo      YOLO Superpixel Project Setup
echo ============================================

:: --- Step 1: Create Project Structure ---
echo.
echo [1/4] Creating directory structure...
mkdir src 2>nul
mkdir src\custom_datasets 2>nul
mkdir src\custom_models 2>nul
mkdir src\training 2>nul
mkdir scripts 2>nul
mkdir models 2>nul
mkdir models\vendor 2>nul
echo Done.

:: --- Step 2: Verify Python Version ---
echo.
echo [2/4] Verifying Python version using 'py.exe' launcher...

:: Check if py.exe exists
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python Launcher 'py.exe' not found.
    echo This script requires the Windows Python Launcher.
    echo Please install Python 3.11.9 from python.org and ensure the launcher is included during installation.
    exit /b 1
)

for /f "tokens=2" %%i in ('py -%REQUIRED_PYTHON_VERSION% --version 2^>^&1') do set PYTHON_VERSION=%%i

if not defined PYTHON_VERSION (
    echo.
    echo ERROR: Python %REQUIRED_PYTHON_VERSION% was not found by 'py.exe'.
    echo Please install Python 3.11.9 and try again.
    exit /b 1
)

echo Found Python version: %PYTHON_VERSION%
if not "%PYTHON_VERSION:~0,4%"=="%REQUIRED_PYTHON_VERSION%" (
    echo.
    echo ERROR: Incorrect Python version detected.
    echo This project requires Python %REQUIRED_PYTHON_VERSION%.x, but 'py.exe' reported %PYTHON_VERSION%.
    exit /b 1
)
echo Python version check passed.

:: --- Step 3: Setup Environment and Install Dependencies ---
echo.
echo [3/4] Setting up environment and installing dependencies...
if not exist "%VENV_NAME%\" (
    echo Creating virtual environment '%VENV_NAME%' with Python %REQUIRED_PYTHON_VERSION%...
    py -%REQUIRED_PYTHON_VERSION% -m venv %VENV_NAME%
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment to install packages...
call "%VENV_NAME%\Scripts\activate.bat"

echo Installing PyTorch with CUDA...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch.
    exit /b 1
)

echo Installing remaining dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)
echo Done.

:: --- Step 4: Clone and Prepare Vendor Repositories ---
echo.
echo [4/4] Cloning and preparing vendor repositories...
:: Handle SPiT
if not exist "models\vendor\SPiT\" (
    echo Cloning SPiT repository...
    git clone %SPIT_REPO_URL% "models\vendor\SPiT"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to clone SPiT repository.
        exit /b 1
    )
    echo Removing .git directory from SPiT...
    rmdir /s /q "models\vendor\SPiT\.git"
) else (
    echo SPiT repository already exists.
)
:: Handle yolov12
if not exist "models\vendor\yolov12\" (
    echo Cloning yolov12 repository...
    git clone %YOLOV12_REPO_URL% "models\vendor\yolov12"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to clone yolov12 repository.
        exit /b 1
    )
    echo Removing .git directory from yolov12...
    rmdir /s /q "models\vendor\yolov12\.git"
) else (
    echo yolov12 repository already exists.
)
echo Done.

echo.
echo ============================================
echo      ✅ Setup Complete!
echo ============================================
echo.
echo Activate the environment with: %VENV_NAME%\Scripts\activate
endlocal

