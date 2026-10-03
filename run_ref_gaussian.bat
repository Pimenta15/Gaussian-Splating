@echo off
REM Roda um script Python do ref-gaussian no ambiente do compilador (CUDA 11.8 + VS 2022),
REM necessario porque nvdiffrast e scene/renderutils compilam extensoes CUDA na primeira execucao.
REM Uso: run_ref_gaussian.bat train.py -s data/ref_nerf/toaster --eval --white_background

if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if not defined PY set "PY=%USERPROFILE%\miniconda3\envs\ref-gaussian\python.exe"
if not defined TORCH_CUDA_ARCH_LIST set TORCH_CUDA_ARCH_LIST=8.9

call "%VCVARS%" >nul || exit /b 1

set "CUDA_PATH=%CUDA_HOME%"
REM pasta do python do ambiente + Scripts (ninja.exe, usado pelas extensoes JIT)
for %%I in ("%PY%") do set "PYDIR=%%~dpI"
set "PATH=%CUDA_HOME%\bin;%PYDIR%;%PYDIR%Scripts;%PYDIR%Library\bin;%PATH%"
set DISTUTILS_USE_SDK=1
REM MSVC 14.4x e mais novo que o suportado oficialmente pelo CUDA 11.8
set "NVCC_APPEND_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
set "CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

cd /d "%~dp0ref-gaussian" || exit /b 1
"%PY%" %*
