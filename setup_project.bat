@echo off
setlocal
cd /d "%~dp0"

echo [Skunkworks] Creating Portable Environment...

:: 1. Redirect ALL temporary space to D: (Fixes "No space left" on C:)
if not exist "D:\temp_install" mkdir "D:\temp_install"
if not exist "D:\pip_cache" mkdir "D:\pip_cache"
set TEMP=D:\temp_install
set TMP=D:\temp_install
set PIP_CACHE_DIR=D:\pip_cache

:: 2. Clean and Create local virtual environment
echo [Skunkworks] Initializing Python 3.10 environment...
if exist "env" (
    echo [Skunkworks] Cleaning existing environment for a fresh start...
    rmdir /s /q env
)
py -3.10 -m venv env

if errorlevel 1 (
    echo [ERROR] Could not create environment. Do you have Python 3.10 installed?
    pause
    exit /b
)

echo [Skunkworks] Environment created. Installing dependencies...

:: 3. Upgrade pip
env\Scripts\python.exe -m pip install --upgrade pip

:: 4. Install PyTorch 2.10.0 with CUDA 12.6 support
echo [Skunkworks] Installing PyTorch 2.10.0 (GPU - cu126)... (This is ~2.5GB, downloading to D:...)
env\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

:: 5. Install UI and Data packages
echo [Skunkworks] Installing UI and Data packages...
env\Scripts\python.exe -m pip install -r requirements.txt

:: 6. Install PyTorch3D (MiroPsota Wheel for Python 3.10 + PyTorch 2.10.0 + CUDA 12.6)
echo [Skunkworks] Installing PyTorch3D (Rendering Engine)...
:: We use a validated community wheel from MiroPsota project compatible with the latest Torch 2.10.0
env\Scripts\python.exe -m pip install "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%%2Bpt2.10.0cu126-cp310-cp310-win_amd64.whl"

if errorlevel 1 (
    echo [ERROR] PyTorch3D failed to install. Please check your internet connection.
    pause
    exit /b
)

echo.
echo [Skunkworks] Setup Complete! 
echo [Skunkworks] Use "run.bat" to start the application.
echo [Skunkworks] Note: All temporary files were safely handled on D:\
pause
endlocal
