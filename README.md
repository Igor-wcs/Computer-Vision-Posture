# Detector de Postura para Home Office - Estudo Comparativo

Este projeto foi desenvolvido como parte de uma **Iniciação Científica / TCC**, com o objetivo de monitorar a postura de usuários em tempo real utilizando visão computacional. O sistema compara dois modelos de estado da arte para detecção de pose: **Google MediaPipe** e **YOLO26n-pose**.

## 🎯 Proposta do Projeto

Com o aumento do trabalho remoto (Home Office), a ergonomia tornou-se uma preocupação central. Este software utiliza a webcam para:
1.  **Calibrar** a postura ideal do usuário.
2.  **Monitorar** em tempo real desvios como:
    *   Cabeça muito baixa (pescoço inclinado).
    *   Ombros desalinhados.
    *   Proximidade excessiva da tela.
    *   Inclinação excessiva para frente.
3.  **Coletar dados** de performance (FPS, CPU, RAM) para fins de análise científica e comparação entre os modelos.

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python 3.11+
*   **Visão Computacional:**
    *   [MediaPipe](https://google.github.io/mediapipe/): Modelo leve e rápido com suporte a coordenadas 3D.
    *   [YOLO26n-pose](https://github.com/ultralytics/ultralytics): Modelo de Deep Learning de alta precisão (CNN).
*   **Interface Gráfica:** CustomTkinter (Visual moderno e responsivo).
*   **Análise de Dados:** Pandas, Seaborn e Matplotlib.
*   **Monitoramento de Sistema:** Psutil.

## 🚀 Como Executar o Projeto

### 1. Preparação do Ambiente
Certifique-se de ter o Python instalado. Recomenda-se o uso de um ambiente virtual:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (Windows)
.\venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executando o Detector com MediaPipe
Este é o modelo mais leve, ideal para uso contínuo em segundo plano.
```bash
python main.py
```

### 3. Executando o Detector com YOLOv26
Este modelo utiliza Deep Learning avançado para maior robustez na detecção.
```bash
python main_yolov26.py
```

### 4. Gerando Gráficos de Comparação Científica
Após realizar testes com ambos os modelos (o que gerará os arquivos `log_mediapipe.csv` e `log_yolo.csv`), você pode gerar a análise visual:
```bash
python generate_graphs.py
```
Isso criará o arquivo `comparativo_modelos.png` e a tabela `resumo_estatistico.csv`.

## 📈 Resultados e Metodologia

O projeto inclui uma metodologia estruturada para Iniciação Científica, detalhada no arquivo `METODOLOGIA_IC.md`. Os testes comparam:
*   **Eficiência:** Quadros por segundo (FPS).
*   **Custo de Hardware:** Porcentagem de uso de CPU e consumo de RAM.
*   **Robustez:** Estabilidade da detecção em diferentes cenários.

---
**Desenvolvido para fins acadêmicos.**
