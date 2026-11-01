# Benchmark de postura: MediaPipe e YOLOv26

Aplicacao Windows para comparar deteccao de postura usando o mesmo roteiro guiado em quatro backends:

- MediaPipe;
- YOLOv26-pose com PyTorch;
- YOLOv26-pose com ONNX Runtime;
- YOLOv26-pose com OpenVINO FP16, exclusivamente na CPU e somente quando houver suporte.

O objetivo e comparar desempenho e estabilidade sob as mesmas poses, camera e condicoes de coleta. O programa nao mede apenas a velocidade do modelo: o FPS registrado representa o comportamento do fluxo de captura e inferencia em tempo real.

## Requisitos

- Windows 10 ou 11, 64 bits;
- Python **3.12** (recomendado: 3.12.x 64 bits);
- webcam acessivel pelo Windows;
- permissao para acessar a camera;
- espaco em disco para as dependencias e os modelos YOLO.

## Instalacao do ambiente

Abra o PowerShell na pasta do projeto e execute:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativacao do ambiente, execute uma vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois, ative novamente o `venv` e instale os requisitos.

## Como executar

O launcher recomendado e `launcher.py`. Para iniciar sem digitar comandos, use:

```text
executar_launcher.bat
```

Tambem e possivel executar diretamente:

```powershell
.\venv\Scripts\Activate.ps1
python launcher.py
```

O launcher detecta as cameras disponiveis e mostra o indice junto do nome fornecido pelo Windows, por exemplo `0 — USB Video Device`. A camera pode ser trocada antes de iniciar um teste.

## Roteiro do usuario

1. Escolha o modelo na lateral esquerda.
2. Escolha a camera, se necessario.
3. Clique em **Calibrar** e mantenha-se **reto** por aproximadamente dois segundos.
4. Clique em **Iniciar teste guiado**. A gravacao do log comeca somente nesse momento.
5. Siga as orientacoes na tela e sustente cada postura ate o sistema avancar.
6. Use **Proximo passo** ou pressione a tecla **espaco** para continuar entre modelos.
7. A ordem padrao e MediaPipe, YOLO/PyTorch, YOLO/ONNX Runtime e, quando habilitado, YOLO/OpenVINO.
8. Ao concluir o ultimo teste, clique em **Gerar relatorio final**.

O botao OpenVINO so aparece quando o modelo IR otimizado consegue ser compilado na CPU. A GPU nao e usada nesse teste. Se a CPU nao tiver suporte, o backend fica indisponivel e os demais testes continuam funcionando.

Para iniciar um experimento independente, use **Arquivar logs/resultados antigos**. Os arquivos sao movidos para `logs/archive/AAAAmmdd-HHMMSS/`; eles nao sao apagados.

## O que esta sendo comparado

Todos os backends recebem frames da mesma camera e executam o mesmo protocolo de postura. O que muda e o executor da inferencia:

| Backend | Modelo/arquivo principal | Execucao |
|---|---|---|
| MediaPipe | Solucao Pose do MediaPipe | Pipeline MediaPipe |
| YOLO/PyTorch | `yolo26n-pose.pt` | Ultralytics/PyTorch |
| YOLO/ONNX Runtime | `yolo26n-pose.onnx` | ONNX Runtime com `CPUExecutionProvider` |
| YOLO/OpenVINO | `yolo26n-pose_openvino_model/yolo26n-pose.xml` e `.bin` | OpenVINO FP16 na CPU |

A comparacao e feita sobre as amostras coletadas durante o mesmo roteiro. Para cada modelo, o programa registra FPS, uso de CPU, RAM, status da postura e coordenadas do nariz. O protocolo pede posturas neutra, inclinacoes laterais, inclinacao para frente, cabeca baixa e afastamento da camera, sempre com retornos a postura correta.

## Metricas e interpretacao

- **FPS:** taxa observada no fluxo de captura e inferencia. A camera pode limitar esse valor; o relatorio tambem informa o limite de FPS detectado/configurado para a camera usada.
- **CPU:** uso do processo relativo ao total de CPUs logicas. Na etapa de relatorio, os valores sao limitados ao intervalo convencional de 0 a 100%; os CSV brutos nao sao alterados.
- **RAM:** memoria residente do processo, em MB.
- **Jitter:** variacao absoluta entre amostras consecutivas da coordenada vertical do nariz; valores menores tendem a indicar maior estabilidade.
- **Calibracao:** define a referencia individual de distancia e largura dos ombros para classificar as poses.

O relatorio identifica a CPU e a RAM do computador que o gera. Se logs forem copiados de outro computador e o relatorio for gerado localmente, essas informacoes representam a maquina local, nao necessariamente a maquina que produziu os logs.

## Arquivos de entrada e saida do relatorio

O relatorio exige estes tres logs:

- `log_mediapipe.csv`;
- `log_yolo.csv`;
- `log_yolo_onnx.csv`.

Se existir e tiver amostras validas, `log_yolo_openvino.csv` tambem e incluido automaticamente.

Os arquivos gerados sao:

- `comparativo_modelos.png`: graficos de FPS, uso de CPU, consumo medio de RAM, jitter e tabelas de minimo, media e maximo;
- `resumo_estatistico.csv`: resumo numerico por modelo e metrica, incluindo desvio padrao e quantidade de amostras;
- `relatorio_benchmark.md`: relatorio textual com hardware, limite de FPS da camera e estatisticas;
- `logs/archive/...`: historico recuperavel de logs e relatorios arquivados.

O codigo que monta o relatorio esta em `reporting.py`. O nucleo de registro e protocolo esta em `benchmark_core.py`; os executores ONNX e OpenVINO ficam em `onnx_pose.py` e `openvino_pose.py`.

## Arquivos principais

- `launcher.py`: interface e fluxo principal;
- `executar_launcher.bat`: inicia o launcher no ambiente Python;
- `benchmark_core.py`: protocolo, classificacao, logs e arquivamento;
- `reporting.py`: leitura dos CSVs, graficos e tabelas;
- `yolo26n-pose.pt`: modelo PyTorch;
- `yolo26n-pose.onnx`: modelo ONNX Runtime;
- `yolo26n-pose_openvino_model/`: modelo IR OpenVINO FP16;
- `GUIA_EXECUCAO.md`: roteiro operacional resumido;
- `METODOLOGIA.md`: detalhes do desenho experimental.