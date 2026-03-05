@echo off
echo =========================================================
echo  Starting Lightweight Federated Learning Experiment
echo =========================================================

echo Checking and installing dependencies...
pip install -r requirements.txt

echo.
echo Running Experiment Pipeline...
python main.py

echo.
echo Experiment Completed. Check EXPERIMENT/ directory for logs.
pause
