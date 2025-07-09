import matplotlib.pyplot as plt
import pandas as pd
import os

# Directorio de salida para gráficos
output_dir_graphics = 'resultados_corridas/riemann_vs_frecuencias_20250708_232216/graficos_modular'
os.makedirs(output_dir_graphics, exist_ok=True)

# Cargar datos de top 5 frecuencias por clase
csv_path = 'resultados_corridas/riemann_vs_frecuencias_20250708_232216/top5_frecuencias_modular.csv'
df = pd.read_csv(csv_path)

# Calcular suma de potencias top 5 para cada clase
sum_potencias = df.groupby(['primo','clase'])['potencia'].sum().reset_index()

# Etiquetas para el gráfico
labels = []
colores = []
for _, row in sum_potencias.iterrows():
    if row['primo'] == 3:
        labels.append(f'N mod 3 = {int(row["clase"])})')
        colores.append(['blue','orange','green'][int(row['clase'])])
    elif row['primo'] == 5:
        labels.append(f'N mod 5 = {int(row["clase"])})')
        colores.append('red')

# Datos para graficar
spectral_energy = sum_potencias['potencia'].values

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, spectral_energy, color=colores)
plt.ylabel('Suma de Potencias Espectrales (Energía Total en Frecuencias Dominantes)')
plt.title('Comparación de Energía Espectral por Clase Modular')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir valores exactos encima de las barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + max(spectral_energy)*0.02, round(yval, 2), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plot_filename = os.path.join(output_dir_graphics, 'comparacion_energia_espectral_modular.png')
plt.savefig(plot_filename)
plt.close()

print(f"Gráfica comparativa de energía espectral guardada en: {plot_filename}")
