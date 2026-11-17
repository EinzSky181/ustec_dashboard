@echo off
title USTEC Breakout Strategy Hub - Web Dashboard
echo ======================================================================
echo           MEMBUKA USTEC BREAKOUT STRATEGY DASHBOARD (WEB UI)
echo ======================================================================
echo.
echo Sedang membuka dashboard web di browser Anda...
echo URL: http://localhost:8501
echo.
echo [INFO] Jangan tutup jendela command prompt ini selama web masih digunakan.
echo        Untuk berhenti, cukup tutup jendela ini atau tekan Ctrl+C.
echo.

start "" http://localhost:8501
python -m streamlit run app.py --server.port=8501

pause
