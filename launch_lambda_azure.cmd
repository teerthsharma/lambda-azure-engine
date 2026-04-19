@echo off
title Lambda Azure Engine - Holographic Terminal
color 0A

echo ===============================================================================
echo                      LAMBDA AZURE ENGINE
echo            Holographic p-adic Subsumption Terminal
echo ===============================================================================
echo.
echo Initializing Grothendieck Context Sheaves...
echo Mapping Q_p ultrametric boundaries...
echo.

cd /d "%~dp0lambda-azure-engine\python-bridge"

if not exist "1_5b_lambda_padic_holographic.bin" (
    echo [ERROR] Packed Holographic weights not found! 
    echo Please run 'python stream_and_quantize_qwen.py' first to build the boundary layer.
    echo.
    pause
    exit /b 1
)

python chat_holographic.py

echo.
echo Holographic projection collapsed.
pause
