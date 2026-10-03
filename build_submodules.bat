@echo off
REM Compila os submodulos CUDA do ref-gaussian no Windows (CUDA 11.8 + VS 2022).
REM Uso: build_submodules.bat [nome-do-submodulo ...]   (sem argumentos = todos)
REM Os caminhos abaixo podem ser sobrescritos por variaveis de ambiente.

if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if not defined PY set "PY=%USERPROFILE%\miniconda3\envs\ref-gaussian\python.exe"
REM RTX 4050 Laptop = compute capability 8.9
if not defined TORCH_CUDA_ARCH_LIST set TORCH_CUDA_ARCH_LIST=8.9

call "%VCVARS%" || exit /b 1

set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
REM MSVC 14.4x e mais novo que o suportado oficialmente pelo CUDA 11.8
set "NVCC_BASE=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
set "CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
REM Extras so do raytracing (usa o Eigen 3.3.7 que o setup.py dele baixa):
REM   --pre-include array : src/bvh.cu usa std::array sem incluir <array> (o STL do MSVC exige)
REM   /Zc:__cplusplus     : sem ele o MSVC reporta __cplusplus=199711L, as conversoes de
REM                         Eigen::half ficam implicitas e __hadd fica ambiguo no CUDA 11.8
set "NVCC_RT=%NVCC_BASE% --pre-include array -Xcompiler /Zc:__cplusplus -Xcompiler /permissive- -UTORCH_API_INCLUDE_EXTENSION_H"

cd /d "%~dp0ref-gaussian" || exit /b 1

set "MODS=%*"
if "%MODS%"=="" set "MODS=simple-knn cubemapencoder diff-surfel-rasterization raytracing"

for %%m in (%MODS%) do (
    echo.
    echo ===== %%m =====
    set "NVCC_APPEND_FLAGS=%NVCC_BASE%"
    if "%%m"=="raytracing" set "NVCC_APPEND_FLAGS=%NVCC_RT%"
    "%PY%" -m pip install --no-build-isolation -v ./submodules/%%m || (echo FALHOU: %%m & exit /b 1)
)
echo.
echo ===== OK =====
