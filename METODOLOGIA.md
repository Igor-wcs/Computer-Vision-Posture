# Metodologia de Pesquisa Experimental

Este documento detalha o delineamento técnico, os fundamentos científicos e o protocolo experimental aplicado para o estudo comparativo entre os modelos de detecção de pose **MediaPipe BlazePose** e **YOLOv26-pose** no contexto de monitoramento ergonômico.

## 1. Delineamento do Método
A pesquisa caracteriza-se como um **estudo experimental quantitativo de natureza aplicada**, utilizando o método de **Análise Comparativa Síncrona**.

### 1.1. Justificativa do Método
A escolha deste método fundamenta-se na necessidade de isolar o desempenho algorítmico como variável independente principal. Ao submeter dois modelos distintos ao mesmo fluxo de entrada (Webcam) e sob protocolos de teste idênticos, é possível quantificar com rigor científico o *trade-off* entre precisão de detecção e custo computacional.

## 2. Materiais e Ambiente de Desenvolvimento

### 2.1. Justificativa de Ferramentas
A seleção das ferramentas baseou-se no equilíbrio entre estabilidade acadêmica e eficiência em tempo real.

| Ferramenta | Versão | Justificativa da Escolha |
| :--- | :--- | :--- |
| **Python** | 3.11 | Versão estável com otimizações de performance e gestão de memória. |
| **MediaPipe** | 0.10.14 | Utiliza a arquitetura *BlazePose*, otimizada para CPUs e dispositivos móveis. |
| **YOLOv26-pose** | 26.x | Representa o estado da arte (SOTA) em Redes Neurais Convolucionais (CNN) para pose. |
| **CustomTkinter** | 5.2.2 | Fornece uma interface moderna com baixo impacto no consumo de recursos. |
| **Psutil** | 5.9.x | Garante medições de CPU e RAM com precisão de nível de sistema. |
| **Pandas/Seaborn** | 2.x / 0.13 | Utilizadas para análise estatística e geração de gráficos científicos. |

### 2.2. Hardware de Referência
Os testes foram executados na seguinte configuração para garantir a reprodutibilidade:
*   **Processador (CPU):** Intel Core Ultra 7 (Arquitetura Lunar Lake)
*   **Memória RAM:** 16 GB
*   **Câmera:** Sensor CMOS 720p @ 30 FPS.
*   **Aceleração:** Processamento executado estritamente via CPU para simular o cenário de Home Office padrão.

## 3. Protocolo Experimental e Coleta de Dados

### 3.1. Coleta de Dados
Dada a natureza de tempo real, optou-se pela **Coleta Própria via Fluxo de Vídeo Direto (Live Capture)**, permitindo avaliar a robustez frente a variações de iluminação e fundo típicas de um ambiente real.

### 3.2. Fases do Experimento
1.  **Fase de Calibração:** O sistema registra as coordenadas base do usuário (posição ideal) durante 2 segundos. Isso normaliza os dados para cada biótipo.
2.  **Sessões de Monitoramento:** Cada modelo é testado em sessões contínuas de aproximadamente **2 minutos**, alternando entre postura ereta e desvios induzidos.
3.  **Log Automático:** O sistema gera arquivos `.csv` registrando métricas a cada segundo, incluindo as coordenadas (X, Y) do nariz e a distância focal para análise de Jitter.

## 4. Métricas de Avaliação
Foram selecionadas quatro métricas fundamentais para cobrir viabilidade técnica e experiência do usuário:

*   **Performance (FPS):** Quantifica a taxa de atualização. Um FPS > 15 é o critério mínimo para monitoramento sem latência perceptível.
*   **Uso de CPU (%):** Mede a eficiência energética. Vital para softwares que rodam em segundo plano.
*   **Consumo de RAM (MB):** Avalia a pegada de memória e compatibilidade com máquinas de entrada.
*   **Estabilidade (Jitter):** Calculada por meio da variância temporal das coordenadas do nariz e distância focal (Head_Dist) capturadas no log durante a postura estática.

## 5. Parâmetros de Análise Postural
A classificação segmenta a postura em cinco estados baseados em cálculos trigonométricos e euclidianos:
1.  **Boa Postura:** Alinhamento nominal entre nariz e ombros.
2.  **Cabeça Baixa:** Desvio vertical (Y) superior a 25% em relação à calibração.
3.  **Ombros Desalinhados:** Assimetria biacromial superior a 5%.
4.  **Inclinado para Frente:** Expansão da largura biacromial projetada superior a 20%.
5.  **Muito Longe:** Retração da largura biacromial superior a 20%.
