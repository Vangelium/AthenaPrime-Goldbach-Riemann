import pandas as pd
import matplotlib.pyplot as plt
import os

# Cargar resultados de correlación
dir_corrida = 'resultados_corridas/riemann_vs_frecuencias_20250708_232216'
archivo_corr = os.path.join(dir_corrida, 'correlacion_frecuencias_riemann_modular.csv')
df = pd.read_csv(archivo_corr)

# Gráfico de dispersión: Frecuencia vs. Δγ/(2π)
plt.figure(figsize=(8, 6))
plt.scatter(df['freq_dominante'], df['diff_riemann/(2pi)'], c=df['primo'], cmap='coolwarm', s=80, edgecolor='k', label=None)
plt.plot([df['freq_dominante'].min(), df['freq_dominante'].max()], [df['freq_dominante'].min(), df['freq_dominante'].max()], 'k--', lw=2, label='y = x')
plt.xlabel('Frecuencia dominante (residuos, clase N mod P = 0)')
plt.ylabel('Δγ / (2π) (diferencia de ceros de Riemann)')
plt.title('Correlación: Frecuencias dominantes vs. diferencias entre ceros de Riemann')
plt.legend(['y = x (correlación perfecta)'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
output_dir = os.path.join(dir_corrida, 'graficos_modular')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'correlacion_frecuencias_riemann_modular.png'))
plt.close()

print(f"Gráfico guardado en: {os.path.join(output_dir, 'correlacion_frecuencias_riemann_modular.png')}")
