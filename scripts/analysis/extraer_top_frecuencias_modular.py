import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

# Cargar datos
df = pd.read_csv('resultados_corridas/riemann_vs_frecuencias_20250708_232216/goldbach_residuos.csv')
primos = [3, 5]
residuos_total = df['Residuo (Total)'].values
N = df['N'].values

resultados_top_freq = []

for p in primos:
    for clase in range(p):
        mask = df[f'N_mod_{p}'] == clase
        residuos_clase = residuos_total[mask]
        N_clase = N[mask]
        if len(residuos_clase) < 32:
            continue
        residuos_centered = residuos_clase - np.mean(residuos_clase)
        N_points = len(residuos_centered)
        espectro = np.abs(fft(residuos_centered))[:N_points // 2]
        frecuencias = fftfreq(N_points, d=1)[:N_points // 2]
        # Extraer top 5 frecuencias (excluyendo la frecuencia cero)
        idxs = np.argsort(espectro[1:])[::-1][:5] + 1  # +1 para saltar el DC
        for rank, idx in enumerate(idxs, 1):
            resultados_top_freq.append({
                'primo': p,
                'clase': clase,
                'rank': rank,
                'frecuencia': frecuencias[idx],
                'potencia': espectro[idx],
                'n': N_points
            })

# Guardar resultados
df_top = pd.DataFrame(resultados_top_freq)
df_top.to_csv('resultados_corridas/riemann_vs_frecuencias_20250708_232216/top5_frecuencias_modular.csv', index=False)

print('Top 5 frecuencias y potencias extraídas por clase. Ver archivo top5_frecuencias_modular.csv')
