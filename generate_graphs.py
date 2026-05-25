import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparative_graphs():
    # Caminhos dos arquivos
    file_mp = 'log_mediapipe.csv'
    file_yolo = 'log_yolo.csv'

    if not os.path.exists(file_mp) or not os.path.exists(file_yolo):
        print("Erro: Arquivos de log não encontrados. Execute os scripts primeiro.")
        return

    # Carregar dados com encoding latin-1 para suportar caracteres especiais nos logs
    df_mp = pd.read_csv(file_mp, encoding='latin-1')
    df_yolo = pd.read_csv(file_yolo, encoding='latin-1')

    # Adicionar coluna de identificação (Forçando nomes padrão para o projeto)
    df_mp['Modelo'] = 'MediaPipe'
    df_yolo['Modelo'] = 'YOLOv26-pose'

    # Concatenar para facilitar plotagem comparativa
    df_combined = pd.concat([df_mp, df_yolo])

    # Configurações de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 10]

    # Criar figura com múltiplos subplots
    fig, axes = plt.subplots(3, 1, sharex=False)
    fig.subplots_adjust(hspace=0.4)

    # 1. Gráfico de FPS
    sns.lineplot(data=df_combined, x=df_combined.index, y='FPS', hue='Modelo', ax=axes[0])
    axes[0].set_title('Comparação de Frames Per Second (FPS)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('FPS')
    axes[0].set_xlabel('Tempo (Frames Processados)')

    # 2. Gráfico de CPU
    sns.lineplot(data=df_combined, x=df_combined.index, y='CPU_Usage_Percent', hue='Modelo', ax=axes[1])
    axes[1].set_title('Uso de CPU (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Uso de CPU %')
    axes[1].set_xlabel('Tempo (Frames Processados)')

    # 3. Gráfico de RAM (Corrigindo o aviso de FutureWarning)
    sns.barplot(data=df_combined, x='Modelo', y='RAM_Usage_MB', hue='Modelo', ax=axes[2], palette='viridis', legend=False)
    axes[2].set_title('Consumo de Memória RAM (Média MB)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('RAM (MB)')


    # Salvar o gráfico combinado
    plt.savefig('comparativo_modelos.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'comparativo_modelos.png' gerado com sucesso!")

    # Gerar tabela de resumo estatístico
    summary = df_combined.groupby('Modelo')[['FPS', 'CPU_Usage_Percent', 'RAM_Usage_MB']].mean().round(2)
    summary.to_csv('resumo_estatistico.csv')
    print("Tabela 'resumo_estatistico.csv' gerada com sucesso!")
    
    print("\n--- RESUMO MÉDIO ---")
    print(summary)

if __name__ == "__main__":
    generate_comparative_graphs()
