import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'DejaVu Sans'

csv_path = os.path.join(os.path.dirname(__file__), 'bioimpedance_dataset.csv')
df = pd.read_csv(csv_path, sep=',', encoding='latin1')

col_zr = "z' / ohm"
col_zi = "z'' / ohm"
col_arquivo = "arquivo"
col_tamanho = "comprimento_total"

maiores = ['semente_05_r2', 'semente_03_r1', 'semente_06_r1', 'semente_04_r2']
menores = ['semente_05_r3', 'semente_08_r2', 'semente_09_r3', 'semente_10_r3']
sementes = maiores + menores
resultados = []

# Regiao de inflexao comum
Z_PRIME_MIN = 25
Z_PRIME_MAX = 30

print("Calculando raio no MESMO ponto de inflexao para todas as sementes (Z' entre 25-30 kOhm):")
print("=" * 120)

for semente in sementes:
    dados = df[df[col_arquivo] == semente].copy()
    if dados.empty:
        continue
    
    tamanho = dados.iloc[0][col_tamanho]
    x = dados[col_zr].abs().values / 1000
    y = dados[col_zi].abs().values / 1000
    
    idx_sort = np.argsort(x)
    x = x[idx_sort]
    y = y[idx_sort]
    
    # Filtrar pontos na regiao de inflexao
    mascara = (x >= Z_PRIME_MIN) & (x <= Z_PRIME_MAX)
    x_inflexao = x[mascara]
    y_inflexao = y[mascara]
    
    if len(x_inflexao) >= 2:
        # Calcular raio como a distancia entre extremos
        raio = (x_inflexao.max() - x_inflexao.min()) / 2
        y_medio = (y_inflexao.min() + y_inflexao.max()) / 2
        
        tipo = 'MAIOR' if semente in maiores else 'MENOR'
        print(f"{tipo} | {semente:20s} ({tamanho:6.1f} cm) | Raio: {raio:6.3f} kOhm | Y_min: {y_inflexao.min():6.2f}, Y_max: {y_inflexao.max():6.2f}")
        
        resultados.append({
            'semente': semente,
            'tamanho': tamanho,
            'raio': raio,
            'x_min': x_inflexao.min(),
            'x_max': x_inflexao.max(),
            'y_min': y_inflexao.min(),
            'y_max': y_inflexao.max(),
            'tipo': tipo
        })
    else:
        tipo = 'MAIOR' if semente in maiores else 'MENOR'
        print(f"{tipo} | {semente:20s} ({tamanho:6.1f} cm) | [SEM DADOS na regiao {Z_PRIME_MIN}-{Z_PRIME_MAX} kOhm]")

df_resultados = pd.DataFrame(resultados)

if len(df_resultados) > 0:
    df_resultados = df_resultados.sort_values('tamanho', ascending=False).reset_index(drop=True)

    # GRAFICO 1: Barras com raios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cores_maiores = plt.cm.Blues(np.linspace(0.4, 0.9, len(maiores)))
    cores_menores = plt.cm.Reds(np.linspace(0.4, 0.9, len(menores)))

    x_pos = np.arange(len(df_resultados))
    cores = []
    for idx, row in df_resultados.iterrows():
        if row['tipo'] == 'MAIOR':
            cores.append(cores_maiores[maiores.index(row['semente'])])
        else:
            cores.append(cores_menores[menores.index(row['semente'])])

    bars = ax1.bar(x_pos, df_resultados['raio'], color=cores, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Sementes (ordenadas por tamanho)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Raio (kOhm)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Raio no Ponto de Inflexao (Z\'={Z_PRIME_MIN}-{Z_PRIME_MAX} kOhm)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{row['tamanho']:.1f}cm" for _, row in df_resultados.iterrows()], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, df_resultados['raio'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # GRAFICO 2: Nyquist com marcacao do ponto de inflexao
    for idx, semente in enumerate(sementes):
        dados = df[df[col_arquivo] == semente]
        tamanho = dados.iloc[0][col_tamanho]
        x = dados[col_zr].abs().values / 1000
        y = dados[col_zi].abs().values / 1000
        
        idx_sort = np.argsort(x)
        x = x[idx_sort]
        y = y[idx_sort]
        
        cor = 'steelblue' if semente in maiores else 'coral'
        estilo = '-' if semente in maiores else '--'
        
        ax2.plot(x, y, estilo, linewidth=2, color=cor, alpha=0.7, label=f'{semente} ({tamanho:.1f}cm)')
        
        # Marcar ponto de inflexao
        mascara = (x >= Z_PRIME_MIN) & (x <= Z_PRIME_MAX)
        if mascara.any():
            x_mark = x[mascara]
            y_mark = y[mascara]
            ax2.plot(x_mark, y_mark, 'o', markersize=8, color=cor, markeredgecolor='black', markeredgewidth=1)

    # Sombrear regiao de inflexao
    ax2.axvspan(Z_PRIME_MIN, Z_PRIME_MAX, alpha=0.2, color='yellow', label=f'Regiao de Inflexao ({Z_PRIME_MIN}-{Z_PRIME_MAX} kOhm)')
    ax2.set_xlabel("Z' (kOhm)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Z'' (kOhm)", fontsize=12, fontweight='bold')
    ax2.set_title('Nyquist - Ponto de Inflexao Marcado', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')
    ax2.set_xlim(0, 40)
    ax2.set_ylim(0, 25)

    plt.tight_layout()
    plt.savefig('analise_raio_inflexao.png', dpi=300, bbox_inches='tight')
    print("\nGrafico salvo como 'analise_raio_inflexao.png'")
    plt.close()

    print("\n" + "="*120)
    print("ESTATISTICAS")
    print("="*120)
    print(f"Maior raio: {df_resultados['raio'].max():.4f} kOhm")
    print(f"Menor raio: {df_resultados['raio'].min():.4f} kOhm")
    print(f"Media: {df_resultados['raio'].mean():.4f} kOhm")
    print(f"Desvio padrao: {df_resultados['raio'].std():.4f} kOhm")

    correlacao = df_resultados['tamanho'].corr(df_resultados['raio'])
    print(f"Correlacao (tamanho vs raio): {correlacao:.4f}")
