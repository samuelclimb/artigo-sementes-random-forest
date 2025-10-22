import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'DejaVu Sans'


# Novas listas de sementes e legenda
maiores = [
    'semente_05_r2',  # 21.6 cm
    'semente_03_r1',  # 19.0 cm
    'semente_06_r1',  # 16.8 cm
]
menores = [
    'semente_05_r3',  # 9.1 cm
    'semente_09_r3',  # 3.9 cm
    'semente_02_r3',  # 2.6 cm
    'semente_08_r2',  # 6.1 cm
]
sementes = maiores + menores

# Paleta científica equilibrada
from matplotlib import cm
cores = [cm.tab10(i) for i in range(10)]

csv_path = os.path.join(os.path.dirname(__file__), 'bioimpedance_dataset.csv')
df = pd.read_csv(csv_path, sep=',', encoding='latin1')

col_zr = "z' / ohm"
col_zi = "z'' / ohm"
col_arquivo = "arquivo"
col_tamanho = "comprimento_total"

fig, ax = plt.subplots(figsize=(10, 7))
labels_tamanhos = []
for idx, semente in enumerate(sementes):
    dados = df[df[col_arquivo] == semente].copy()
    if dados.empty:
        continue
    # Ordena por frequência
    if 'freq / hz' in dados.columns:
        dados = dados.sort_values('freq / hz')
    tamanho = dados.iloc[0][col_tamanho]
    estilo = '-' if semente in maiores else '--'
    # Garante primeiro quadrante
    x = dados[col_zr].abs() / 1000  # em kΩ
    y = dados[col_zi].abs() / 1000  # em kΩ
    line, = ax.plot(
        x,
        y,
        linestyle=estilo,
        color=cores[idx],
        linewidth=1.8,
        label=f"{tamanho} cm"
    )
    # store (handle, label, numeric_size) so we can sort legend entries by length
    labels_tamanhos.append((line, f"{tamanho} cm", tamanho))
ax.set_xlabel("Z' (kΩ)", fontsize=14)
ax.set_ylabel("–Z'' (kΩ)", fontsize=14)
ax.set_xlim(0, 40)
ax.set_ylim(0, 25)
# Title removed per request (keep plot without a title)
ax.grid(True, alpha=0.3)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
labels_tamanhos.sort(key=lambda x: x[2], reverse=True)
handles = [lt[0] for lt in labels_tamanhos]
labels = [lt[1] for lt in labels_tamanhos]
# Place legend inside axes at upper-left corner (axes coordinates)
leg = ax.legend(handles, labels, title="Comprimento (cm)", fontsize=11, title_fontsize=12,
                loc='upper left', bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes,
                frameon=True, framealpha=0.9)
for lh in leg.legend_handles:
    lh.set_linewidth(3)
plt.tight_layout()
plt.savefig('nyquist_maiores_vs_menores_40kx40k.png', dpi=300)
plt.show()
print('Gráfico de Nyquist salvo como nyquist_maiores_vs_menores_40kx40k.png')
