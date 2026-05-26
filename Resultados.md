📊 Resultados do Estudo Comparativo

Este documento apresenta os dados consolidados obtidos durante a execução dos testes comparativos controlados (sessões de aproximadamente 2 min) entre os modelos MediaPipe (BlazePose) e YOLOv26-pose.

## 1. Resumo Quantitativo de Performance
Os resultados demonstram um clássico *trade-off* computacional entre eficiência de hardware e precisão estática.

| Métrica | MediaPipe (BlazePose) | YOLOv26 (Nano-pose) | Comparação Direta |
| :--- | :--- | :--- | :--- |
| **FPS Médio** | 21.39 (±2.25) | 11.44 (±1.26) | MediaPipe é ~87% mais rápido |
| **Uso de CPU (%)** | 38.44% | 98.71% | YOLO causa gargalo (satura a CPU) |
| **Uso de RAM (MB)** | 290.7 MB | 504.2 MB | YOLO consome ~73% mais memória |
| **Estabilidade (Jitter)** | 0.0161 | 0.0107 | YOLO "treme" menos (maior precisão) |

*(Nota: O Jitter reflete a variância frame a frame da coordenada Y do nariz. Valores menores indicam maior estabilidade da detecção).*

## 2. Análise Técnica e Discussão do *Trade-off*

### 2.1. O Gargalo Computacional do YOLOv26
Os dados confirmam que a arquitetura CNN do YOLOv26-pose, executada estritamente via CPU em um processador moderno (Intel Core Ultra), resulta em saturação quase total dos núcleos (**98.7%**). Devido a este alto custo, o modelo não consegue atingir o limiar aceitável de 15 FPS definido na metodologia, operando a uma média restritiva de **11.4 FPS**. Isso inviabiliza o YOLOv26 como uma solução primária para execução em *background* sem aceleração de hardware (GPU dedicada).

### 2.2. A Eficiência do MediaPipe
O MediaPipe cumpriu com os requisitos metodológicos para um monitoramento fluido, mantendo **21.3 FPS** com uma margem segura de uso de CPU (**38.4%**). Essa folga de processamento garante que o computador do usuário possa executar outras ferramentas (como chamadas de vídeo em Home Office) sem impacto severo na experiência.

### 2.3. O Fator de Precisão (Jitter)
Apesar do alto custo, o YOLOv26 obteve uma vitória científica expressiva na **Estabilidade de Coordenadas**. Com um índice de Jitter de **0.0107** contra **0.0161** do MediaPipe, o YOLO demonstra ser matematicamente menos suscetível a tremores algorítmicos. Em cenários estáticos, as predições do YOLO oscilam consideravelmente menos.

## 3. Conclusões Experimentais
O experimento valida a hipótese de que o **MediaPipe é a solução mais escalável e viável** para democratização da ergonomia em computadores pessoais, pois cumpre o limiar de performance necessário. No entanto, o estudo comprova cientificamente que, se a latência e o consumo de hardware não forem impeditivos (ex: máquinas dedicadas ou equipadas com GPU), o **YOLOv26 oferece uma detecção mais firme e confiável (menor Jitter)**.
