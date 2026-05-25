# Metodologia Detalhada da Pesquisa Experimental

Este documento descreve os fundamentos científicos, a escolha dos materiais e o método experimental aplicado para a comparação entre os modelos de detecção de pose em aplicações de ergonomia.

## 1. Justificativa de Materiais e Ferramentas

A seleção das ferramentas baseou-se no equilíbrio entre **modernidade acadêmica**, **estabilidade de bibliotecas** e **acessibilidade de hardware**.

| Ferramenta | Versão | Justificativa da Escolha |
| :--- | :--- | :--- |
| **Python** | 3.11 | Versão estável com otimizações de performance em relação a versões anteriores, essencial para processamento de vídeo. |
| **MediaPipe** | 0.10.14 | Escolhido por utilizar a arquitetura *BlazePose*, que permite detecção de landmarks 3D em tempo real mesmo em dispositivos sem GPU dedicada. |
YOLOv26 (Ultralytics) | 26.x | Versão de estado da arte (SOTA) em 2026 para detecção de pose.
| **CustomTkinter** | 5.2.2 | Utilizada para prover uma interface de usuário (UI) moderna (estilo Material Design), garantindo que o foco do usuário seja a postura, sem distrações visuais datadas. |
| **Psutil** | 5.9.x | Ferramenta padrão da indústria para monitoramento de processos, permitindo medições de CPU e RAM com precisão de nível de sistema operacional. |
| **Pandas/Seaborn** | 2.x / 0.13 | Escolhidas pela capacidade de manipulação estatística e geração de gráficos de nível científico (ready-for-publication). |

## 2. Metodologia do Experimento

O método adotado é a **Pesquisa Quantitativa Experimental Comparativa**. 

### Por que este método?
A escolha deve-se à necessidade de isolar a variável "Modelo de Visão Computacional" e observar como ela afeta as variáveis dependentes (Desempenho e Consumo). O experimento segue o protocolo de **ambiente controlado**:
1.  **Isolamento de Variáveis:** Ambos os modelos processam o mesmo fluxo de entrada (Webcam padrão) sob as mesmas condições de iluminação.
2.  **Reprodutibilidade:** Através da calibração inicial, garante-se que os limiares de "Boa Postura" sejam normalizados para o mesmo usuário antes da coleta de dados.
3.  **Coleta Passiva:** O log é gerado em tempo real sem interferência humana, eliminando viés de observação na coleta dos dados técnicos.

## 3. Descrição das Métricas Aplicadas

Para validar a viabilidade técnica dos modelos, foram definidas quatro métricas fundamentais:

### A. Performance de Inferência (FPS)
*   **O que mede:** A quantidade de quadros que o sistema processa por segundo.
*   **Importância:** Determina a "fluidez" do monitoramento. Baixo FPS pode causar atrasos (lag) nos alertas, tornando a ferramenta ineficaz para correções em tempo real.

### B. Custo Computacional (Uso de CPU %)
*   **O que mede:** A carga de processamento exigida do processador principal.
*   **Importância:** Como o software de postura deve rodar *background* enquanto o usuário trabalha, um uso de CPU elevado (acima de 50-60%) é considerado proibitivo para uso comercial/doméstico.

### C. Eficiência de Memória (RAM Usage MB)
*   **O que mede:** A pegada (footprint) de memória RAM do processo Python.
*   **Importância:** Avalia se o modelo possui vazamentos de memória (memory leaks) e se é compatível com computadores com pouca memória (ex: 8GB RAM).

### D. Confiabilidade Postural (Status Log)
*   **O que mede:** A frequência de troca entre estados (ex: Boa Postura -> Cabeça Baixa).
*   **Importância:** Ajuda a identificar o "jitter" (ruído) do modelo. Um modelo que troca de status muitas vezes em 1 segundo é menos confiável do que um modelo estável.

## 4. Procedimento de Teste
1.  Inicialização do sistema e calibração da postura ereta (2 segundos).
2.  Execução de 5 minutos de monitoramento contínuo por modelo.
3.  Simulação de 3 cenários de erro postural (Cabeça baixa, Ombros desalinhados, Proximação).
4.  Geração do gráfico comparativo e análise das médias estatísticas.
