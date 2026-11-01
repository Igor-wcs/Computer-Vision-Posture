# Guia do benchmark padronizado

## Ambiente

Use Python 3.12. O MediaPipe deste projeto nao deve ser executado com Python 3.14.

```powershell
python -m pip install -r requirements.txt
python launcher.py
```

## Antes de um experimento novo

Abra o launcher e consulte o indicador de logs. Os logs sao aditivos. Use **Arquivar logs/resultados antigos** para mover logs e relatorios anteriores a uma pasta recuperavel em `logs/archive/`.

## Roteiro automatizado

Execute MediaPipe, YOLOv26/PyTorch e depois YOLOv26/ONNX Runtime. Se o botão estiver disponível, execute também YOLOv26/OpenVINO FP16. Em cada modelo:

1. Fique inteiramente visivel, sente-se reto e calibre por 2 segundos.
2. Inicie o teste. O cronometro e a gravacao comecam juntos.
3. Siga a sequencia exibida: neutra, inclinacao esquerda, neutra, inclinacao direita, neutra, frente, neutra, cabeca baixa, neutra, afastado e neutra.
4. Sustente cada postura ate o programa reconhecer e avancar.
5. Ao final, a gravacao e o cronometro param automaticamente e o resumo da sessao aparece.

Use **Testar novamente** para outra sessao com o mesmo modelo ou **Proximo teste** para continuar. O relatorio so e gerado quando os dois logs existem e possuem amostras validas.

## Saidas

- `log_mediapipe.csv`, `log_yolo.csv`, `log_yolo_onnx.csv` e, quando disponível, `log_yolo_openvino.csv`: logs aditivos coletados somente durante o roteiro.
- `comparativo_modelos.png`: quatro graficos e identificacao de CPU/RAM.
- `resumo_estatistico.csv`: minimo, media, maximo, desvio padrao e numero de amostras.
- `relatorio_benchmark.md`: resumo tecnico em texto.

O teste ONNX Runtime executa `yolo26n-pose.onnx` diretamente com ONNX Runtime. OpenVINO usa o modelo IR FP16 otimizado exclusivamente na CPU; a GPU nao e utilizada. O launcher libera este ultimo somente apos verificar a instalacao e a compilacao do modelo na CPU. O relatorio inclui OpenVINO quando seu log existe.
