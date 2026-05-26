# Detector de Postura para Home Office - Estudo Comparativo

Este projeto utiliza Visão Computacional para monitorar a postura de usuários em tempo real, comparando dois modelos de ponta: **Google MediaPipe** e **YOLOv26-pose**.

## 🎯 Proposta do Projeto
Com o foco em ergonomia no trabalho remoto, este software utiliza a webcam para:
1.  **Calibrar** a postura ideal personalizada.
2.  **Monitorar** desvios (cabeça baixa, ombros desalinhados, inclinação).
3.  **Coletar dados** científicos de performance (FPS, CPU, RAM).

## 🛠️ Tecnologias Utilizadas
*   **Linguagem:** Python 3.11+
*   **IA/Visão:** MediaPipe & YOLOv26 (Ultralytics).
*   **Interface:** CustomTkinter.
*   **Análise:** Pandas, Seaborn, Matplotlib & Psutil.

## 📂 Organização do Projeto
*   `main.py`: Execução via MediaPipe (Versão Leve).
*   `main_yolov26.py`: Execução via YOLOv26 na versão nano (Versão mais leve de Alta Precisão).
*   `generate_graphs.py`: Script para gerar análises e gráficos comparativos.
*   `METODOLOGIA.md`: Detalhamento do método científico e métricas.
*   `RESULTADOS.md`: Relatório comparativo de performance e eficiência.

## 🚀 Como Executar
1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute um dos modelos:**
    ```bash
    python main.py           # Para MediaPipe
    python main_yolov26.py   # Para YOLO
    ```
3.  **Gere os gráficos (após os testes):**
    ```bash
    python generate_graphs.py
    ```

---
*Desenvolvido como projeto de Iniciação Científica / TCC.*
