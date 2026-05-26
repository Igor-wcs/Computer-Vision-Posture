import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparative_graphs():
    # Caminhos dos arquivos
    file_mp = 'log_mediapipe.csv'
    file_yolo = 'log_yolo.csv'

    if not os.path.exists(file_mp) or not os.path.exists(file_yolo):
        print("Erro: Arquivos de log não encontrados. Execute os scripts primeiro para gerar novos dados.")
        return

    # Carregar dados
    df_mp = pd.read_csv(file_mp, encoding='latin-1')
    df_yolo = pd.read_csv(file_yolo, encoding='latin-1')

    # Identificação
    df_mp['Modelo'] = 'MediaPipe'
    df_yolo['Modelo'] = 'YOLOv26'

    # Concatenar
    df_combined = pd.concat([df_mp, df_yolo])

    # Cálculo de Jitter (Variância em uma janela móvel para capturar instabilidade)
    # Vamos focar no Nose_Y e Head_Dist que são críticos para a postura
    df_mp['Jitter_Nose_Y'] = df_mp['Nose_Y'].diff().abs()
    df_yolo['Jitter_Nose_Y'] = df_yolo['Nose_Y'].diff().abs()
    
    # Configurações de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 14]

    # Criar figura com 4 subplots
    fig, axes = plt.subplots(4, 1, sharex=False)
    fig.subplots_adjust(hspace=0.5)

    # 1. Gráfico de FPS
    sns.lineplot(data=df_combined, x=df_combined.index, y='FPS', hue='Modelo', ax=axes[0])
    axes[0].set_title('1. Performance: Frames Per Second (FPS)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('FPS')

    # 2. Gráfico de CPU
    sns.lineplot(data=df_combined, x=df_combined.index, y='CPU_Usage_Percent', hue='Modelo', ax=axes[1])
    axes[1].set_title('2. Eficiência: Uso de CPU (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('CPU %')

    # 3. Gráfico de RAM
    sns.barplot(data=df_combined, x='Modelo', y='RAM_Usage_MB', hue='Modelo', ax=axes[2], palette='viridis', legend=False)
    axes[2].set_title('3. Memória: Consumo de RAM (MB)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('RAM (MB)')

    # 4. Gráfico de Jitter (Estabilidade das Coordenadas)
    # Usamos um Boxplot para mostrar a dispersão do "tremor" nas coordenadas
    df_jitter = pd.concat([
        df_mp[['Modelo', 'Jitter_Nose_Y']], 
        df_yolo[['Modelo', 'Jitter_Nose_Y']]
    ]).dropna()
    
    sns.boxplot(data=df_jitter, x='Modelo', y='Jitter_Nose_Y', hue='Modelo', ax=axes[3], palette='magma', legend=False)
    axes[3].set_title('4. Estabilidade: Jitter (Tremor das Coordenadas do Nariz)', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Variação entre Frames')
    axes[3].set_yscale('log') # Escala logarítmica para melhor visualização de pequenas variações

    # Salvar o gráfico combinado
    plt.savefig('comparativo_modelos.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'comparativo_modelos.png' (4 painéis) gerado com sucesso!")

    # Gerar tabela de resumo estatístico expandida
    metrics = ['FPS', 'CPU_Usage_Percent', 'RAM_Usage_MB', 'Nose_Y', 'Head_Dist']
    summary = df_combined.groupby('Modelo')[metrics].agg(['mean', 'std']).round(4)
    
    # Adicionar a média do Jitter manualmente
    summary.loc['MediaPipe', ('Jitter', 'mean')] = df_mp['Jitter_Nose_Y'].mean()
    summary.loc['YOLOv26', ('Jitter', 'mean')] = df_yolo['Jitter_Nose_Y'].mean()

    summary.to_csv('resumo_estatistico.csv')
    print("Tabela 'resumo_estatistico.csv' com métricas de Jitter gerada com sucesso!")
    
    print("\n--- RESUMO ESTATÍSTICO CIENTÍFICO ---")
    print(summary)

if __name__ == "__main__":
    generate_comparative_graphs()
