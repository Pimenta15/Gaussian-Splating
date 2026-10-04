@echo off
REM Treina e avalia em fila as cenas do Shiny Blender Synthetic (argumentos do train.sh).
REM Logs em ref-gaussian\logs\<cena>_train.txt e <cena>_eval.txt; metricas em output\<cena>\metric.txt
set "RUN=%~dp0run_ref_gaussian.bat"
set "LOGS=%~dp0ref-gaussian\logs"
if not exist "%LOGS%" mkdir "%LOGS%"

call :cena coffee
call :cena helmet --lambda_normal_smooth 1.0
call :cena ball --lambda_normal_smooth 1.0
call :cena teapot
call :cena car
echo %date% %time% FILA CONCLUIDA>> "%LOGS%\queue.txt"
exit /b 0

:cena
set "C=%~1"
shift
set "EXTRA="
:junta
if "%~1"=="" goto roda
set "EXTRA=%EXTRA% %1"
shift
goto junta
:roda
if exist "%~dp0ref-gaussian\output\%C%\metric.txt" (
    echo %date% %time% %C% ja avaliada, pulando>> "%LOGS%\queue.txt"
    exit /b 0
)
echo %date% %time% %C% treino inicio>> "%LOGS%\queue.txt"
call "%RUN%" train.py -s data/ref_nerf/%C% -m output/%C% --eval --white_background%EXTRA% > "%LOGS%\%C%_train.txt" 2>&1
if errorlevel 1 (
    echo %date% %time% %C% TREINO FALHOU>> "%LOGS%\queue.txt"
    exit /b 0
)
echo %date% %time% %C% avaliacao inicio>> "%LOGS%\queue.txt"
call "%RUN%" eval.py --white_background --save_images --model_path output/%C% > "%LOGS%\%C%_eval.txt" 2>&1
if errorlevel 1 (echo %date% %time% %C% AVALIACAO FALHOU>> "%LOGS%\queue.txt") else (echo %date% %time% %C% concluida>> "%LOGS%\queue.txt")
exit /b 0
