📊 Relatório Atualizado de Comparação (Dados Recentes)

  1. Performance de Hardware e Processamento
  Houve uma estabilização nos resultados, mostrando que o MediaPipe é a opção mais leve para processamento
  em tempo real.

  ┌──────────────────┬───────────────────────┬────────────────────┬─────────────────────────────┐
  │ Métrica          │ MediaPipe (BlazePose) │ YOLOv26 (Nano-pose) │ Comparação                  │
  ├──────────────────┼───────────────────────┼────────────────────┼─────────────────────────────┤
  │ FPS Médio        │ 20.88 FPS             │ 13.06 FPS          │ MediaPipe é 60% mais rápido │
  │ Uso de CPU Médio │ 21.2%                 │ 91.8%              │ YOLO exige 4.3x mais CPU    │
  │ Uso de RAM Médio │ 291.2 MB              │ 504.6 MB           │ YOLO consome 73% mais RAM   │
  └──────────────────┴───────────────────────┴────────────────────┴─────────────────────────────┘

  2. Análise Detalhada dos Novos Dados

   * Eficiência Energética e Multitarefa: O MediaPipe manteve um consumo de CPU extremamente baixo e estável (entre 16%
     e 26%), o que é ideal para o público de Home Office que precisa de recursos sobrando para chamadas de vídeo e
     outras aplicações. 
   * Gargalo de CPU (YOLOv26): O YOLOv26 operou quase constantemente acima de 90% de uso de CPU, chegando a atingir 100%
     de uso em vários momentos (ex: 18:13:00). Isso indica que o modelo está limitado pelo processador do sistema e
     poderia se beneficiar imensamente de uma GPU (placa de vídeo) para aliviar o processamento.
   * Consistência de Detecção: 
       * No MediaPipe, os logs mostram uma transição suave entre "Boa Postura" e "Inclinado para Frente", sugerindo que
         as coordenadas 3D (Z) ajudam a criar uma zona de transição mais clara.
  3. Conclusão para Iniciação Científica
  Os novos dados reforçam a tese de que o MediaPipe é a solução mais viável para o usuário final comum devido à sua
  economia de recursos. Por outro lado, o YOLOv26 prova ser uma ferramenta de detecção mais "robusta" e analítica, sendo recomendada para aplicações onde a máquina é dedicada exclusivamente à tarefa ou possui hardware de aceleração (GPU).

  ---