# Metodologia de Comparação: MediaPipe vs YOLOv26-pose

Para uma Iniciação Científica (IC), a comparação não deve ser apenas "qual é melhor", mas sim **"em quais condições e sob quais métricas cada um se destaca"**. Aqui está uma estrutura de análise:

## 1. Métricas de Desempenho (Performance)
*   **Latência / FPS (Frames Per Second):** Medir quantos quadros por segundo cada modelo processa na mesma máquina.
*   **Consumo de Recursos:** Utilizar bibliotecas como `psutil` para monitorar o uso de CPU (%) e Memória RAM (MB) durante a execução de cada script.
*   **Tempo de Inicialização:** Quanto tempo o modelo leva para carregar na memória e estar pronto para a primeira inferência.

## 2. Métricas de Precisão (Accuracy)
Como não temos um "ground truth" (verdade absoluta) automático, você pode:
*   **Erro Médio Absoluto (MAE):** Gravar um vídeo de um usuário fazendo movimentos controlados e comparar a oscilação das coordenadas dos ombros entre os dois modelos. O modelo com menor "jitter" (tremor nas coordenadas) geralmente é mais estável.
*   **Taxa de Sucesso na Calibração:** Quantas vezes o modelo falha em detectar os pontos necessários para iniciar o sistema em diferentes distâncias.

## 3. Análise de Robustez (Estresse do Modelo)
Crie cenários controlados para testar o limite de cada um:
*   **Luminosidade:** Testar em ambiente bem iluminado vs ambiente escuro.
*   **Oclusão Parcial:** O que acontece quando o usuário coloca a mão na frente do rosto ou quando parte do ombro sai do quadro?
*   **Distância:** Testar a precisão a 50cm, 1m e 2m da câmera.
*   **Background complexo:** Testar com fundo neutro vs fundo com muitas pessoas ou objetos.

## 4. Sugestão de Ferramenta para Log de Dados
Você pode adicionar um pequeno código nos seus scripts para salvar um arquivo `.csv` com os dados:
```python
import time
import csv

# Exemplo de log para IC
with open('benchmarks.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    # [Modelo, Timestamp, FPS, CPU_Usage, Mem_Usage, Detecção_Sucesso]
    writer.writerow(['YOLOv8', time.time(), fps, cpu, mem, True])
```

## 5. Estrutura do Trabalho Acadêmico
1.  **Introdução:** Contexto de Home Office e Ergonomia.
2.  **Referencial Teórico:** Como funciona a Pose Estimation (MediaPipe BlazePose vs YOLOv8 CNN).
3.  **Metodologia:** Descrição do hardware usado e dos cenários de teste.
4.  **Resultados:** Gráficos comparativos (FPS, Estabilidade, Erro).
5.  **Conclusão:** Indicação de qual modelo é melhor para cada perfil de hardware (ex: MediaPipe para PCs fracos, YOLOv8 para maior precisão em ambientes variados).
