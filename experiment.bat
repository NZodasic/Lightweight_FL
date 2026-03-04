@echo off
setlocal enabledelayedexpansion

:: experiment.bat
:: ==============
:: Convenience script to run all sparsity x client configurations
:: from the Lightweight-Fed-NIDS paper (Bouayad et al., 2024).
::
:: Paper Table II & III: sparsity=[0%,50%,70%,90%] x clients=[10,50,100]
::
:: Usage:
::   experiment.bat
::   experiment.bat non_iid

set "PARTITION=iid"
if not "%~1"=="" set "PARTITION=%~1"

set "RAW_DATA=data\USTC-TFC2016"
set "PROC_DATA=data\processed"
set "IMAGE_SIZE=224"
set "ROUNDS=5"
set "LOCAL_EPOCHS=10"
set "LR=0.0002"
set "SEED=42"

echo ========================================================
echo  Lightweight-Fed-NIDS Full Experiment
echo  Dataset   : USTC-TFC2016
echo  Model     : ResNet-50
echo  Partition : %PARTITION%
echo ========================================================

for %%C in (10 50 100) do (
  for %%S in (0.0 0.5 0.7 0.9) do (
    echo.
    echo --------------------------------------------------------
    echo  Clients=%%C, Sparsity=%%S
    echo --------------------------------------------------------
    python main.py ^
      --raw_data "%RAW_DATA%" ^
      --proc_data "%PROC_DATA%" ^
      --image_size %IMAGE_SIZE% ^
      --num_clients %%C ^
      --sparsity %%S ^
      --rounds %ROUNDS% ^
      --local_epochs %LOCAL_EPOCHS% ^
      --lr %LR% ^
      --partition %PARTITION% ^
      --seed %SEED%
  )
)

echo.
echo ========================================================
echo  All experiments complete. Results in EXPERIMENT/ folder
echo ========================================================
