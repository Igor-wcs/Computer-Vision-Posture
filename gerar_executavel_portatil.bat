@echo off
setlocal EnableExtensions

rem Gera um unico executavel portatil em .\portable\BenchmarkPostura.exe.
rem Execute este arquivo dentro da pasta do projeto, apos atualizar o codigo.

cd /d "%~dp0"
set "PROJECT_DIR=%CD%"
set "PYTHON=%PROJECT_DIR%\venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo ERRO: nao foi encontrado o ambiente virtual em "%PROJECT_DIR%\venv".
    echo Crie o venv e instale os requisitos antes de gerar o executavel.
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\launcher.py" (
    echo.
    echo ERRO: launcher.py nao foi encontrado nesta pasta.
    pause
    exit /b 1
)

for %%F in ("yolo26n-pose.pt" "yolo26n-pose.onnx" "yolo26n-pose_openvino_model\yolo26n-pose.xml" "yolo26n-pose_openvino_model\yolo26n-pose.bin") do (
    if not exist "%PROJECT_DIR%\%%~F" (
        echo.
        echo ERRO: recurso obrigatorio ausente: %%~F
        pause
        exit /b 1
    )
)

echo.
echo Verificando o PyInstaller...
"%PYTHON%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo Instalando o PyInstaller no ambiente de compilacao...
    "%PYTHON%" -m pip install "pyinstaller==6.19.0"
    if errorlevel 1 (
        echo.
        echo ERRO: nao foi possivel instalar o PyInstaller.
        pause
        exit /b 1
    )
)

for /f "usebackq delims=" %%I in (`"%PYTHON%" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"`) do set "SITE_PACKAGES=%%I"
set "TORCHVISION_DIR=%SITE_PACKAGES%\torchvision"
set "TORCHVISION_EXTENSION=%TORCHVISION_DIR%\_C_stable.pyd"
if not exist "%TORCHVISION_EXTENSION%" set "TORCHVISION_EXTENSION=%TORCHVISION_DIR%\_C.pyd"

if not exist "%TORCHVISION_EXTENSION%" (
    echo.
    echo ERRO: a extensao nativa do Torchvision nao foi encontrada.
    echo Reinstale torch e torchvision no venv antes de gerar o executavel.
    pause
    exit /b 1
)

echo.
echo Gerando o executavel portatil. Isso pode levar alguns minutos...
"%PYTHON%" -m PyInstaller --noconfirm --clean --onefile --windowed --name BenchmarkPostura ^
  --distpath "%PROJECT_DIR%\portable" ^
  --workpath "%PROJECT_DIR%\build_portable" ^
  --specpath "%PROJECT_DIR%\build_specs" ^
  --add-data "%PROJECT_DIR%\yolo26n-pose.pt;." ^
  --add-data "%PROJECT_DIR%\yolo26n-pose.onnx;." ^
  --add-data "%PROJECT_DIR%\yolo26n-pose_openvino_model;yolo26n-pose_openvino_model" ^
  --add-binary "%TORCHVISION_EXTENSION%;torchvision" ^
  --add-binary "%TORCHVISION_DIR%\image_stable.pyd;torchvision" ^
  --add-binary "%TORCHVISION_DIR%\jpeg8.dll;torchvision" ^
  --add-binary "%TORCHVISION_DIR%\libpng16.dll;torchvision" ^
  --add-binary "%TORCHVISION_DIR%\libsharpyuv.dll;torchvision" ^
  --add-binary "%TORCHVISION_DIR%\libwebp.dll;torchvision" ^
  --add-binary "%TORCHVISION_DIR%\zlib.dll;torchvision" ^
  --hidden-import torchvision ^
  --hidden-import torchvision.extension ^
  --hidden-import torchvision.ops ^
  --collect-all customtkinter ^
  --collect-all mediapipe ^
  --collect-all ultralytics ^
  --collect-all openvino ^
  --collect-all onnxruntime ^
  "%PROJECT_DIR%\launcher.py"

if errorlevel 1 (
    echo.
    echo ERRO: a geracao do executavel falhou. Consulte as mensagens acima.
    pause
    exit /b 1
)

echo.
echo Concluido: "%PROJECT_DIR%\portable\BenchmarkPostura.exe"
echo Copie somente esse arquivo para usar o programa de forma portatil.
pause
endlocal
