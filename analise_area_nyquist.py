import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import trapz

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

print("Calculando AREA sob a curva de Nyquist para cada semente:")
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
    
    # Calcular area usando metodo trapezoidal
    area = trapz(y, x)
    
    tipo = 'MAIOR' if semente in maiores else 'MENOR'
    print(f"{tipo} | {semente:20s} ({tamanho:6.1f} cm) | Area: {area:8.2f} kOhm²")
    
    resultados.append({
        'semente': semente,
        'tamanho': tamanho,
        'area': area,
        'tipo': tipo,
        'x_data': x,
        'y_data': y
    })

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('tamanho', ascending=False).reset_index(drop=True)

# GRAFICO
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

# Grafico 1: Areas
bars = ax1.bar(x_pos, df_resultados['area'], color=cores, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Sementes (ordenadas por tamanho)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Area sob a Curva (kOhm²)', fontsize=12, fontweight='bold')
ax1.set_title('Area da Curva de Nyquist por Semente', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"{row['tamanho']:.1f}cm" for _, row in df_resultados.iterrows()], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, df_resultados['area'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val:.0f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Grafico 2: Nyquist com areas preenchidas
for idx, row in df_resultados.iterrows():
    semente = row['semente']
    x = row['x_data']
    y = row['y_data']
    
    cor = 'steelblue' if row['tipo'] == 'MAIOR' else 'coral'
    estilo = '-' if row['tipo'] == 'MAIOR' else '--'
    
    ax2.plot(x, y, estilo, linewidth=2.5, color=cor, alpha=0.8, label=f'{semente} ({row["tamanho"]:.1f}cm, A={row["area"]:.0f})')
    ax2.fill_between(x, 0, y, alpha=0.1, color=cor)

ax2.set_xlabel("Z' (kOhm)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Z'' (kOhm)", fontsize=12, fontweight='bold')
ax2.set_title('Nyquist - Areas sob as Curvas Preenchidas', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc='best', ncol=2)
ax2.set_xlim(0, 220)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('analise_area_nyquist.png', dpi=300, bbox_inches='tight')
print("\nGrafico salvo como 'analise_area_nyquist.png'")
plt.close()

print("\n" + "="*120)
print("ESTATISTICAS")
print("="*120)
print(f"Maior area: {df_resultados['area'].max():.2f} kOhm²")
print(f"Menor area: {df_resultados['area'].min():.2f} kOhm²")
print(f"Media: {df_resultados['area'].mean():.2f} kOhm²")
print(f"Desvio padrao: {df_resultados['area'].std():.2f} kOhm²")

correlacao = df_resultados['tamanho'].corr(df_resultados['area'])
print(f"\nCorrelacao (tamanho vs area): {correlacao:.4f}")
if correlacao > 0.5:
    print("Interpretacao: Forte CORRELACAO POSITIVA = sementes maiores tem MAIS area (energia dissipada)")
elif correlacao < -0.5:
    print("Interpretacao: Forte CORRELACAO NEGATIVA = sementes menores tem MAIS area")
else:
    print("Interpretacao: Correlacao FRACA")
