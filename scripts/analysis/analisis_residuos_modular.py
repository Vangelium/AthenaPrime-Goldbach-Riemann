import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Cargar datos
df = pd.read_csv('resultados_corridas/riemann_vs_frecuencias_20250708_232216/goldbach_residuos.csv')

# Parámetros de análisis
primos = [3, 5]
residuos_total = df['Residuo (Total)'].values
N = df['N'].values

resultados_estadisticos = {}
resultados_frecuencias = {}

for p in primos:
    for clase in range(p):
        mask = df[f'N_mod_{p}'] == clase
        residuos_clase = residuos_total[mask]
        N_clase = N[mask]

        if len(residuos_clase) < 10:
            continue

        # Estadísticas descriptivas
        media = np.mean(residuos_clase)
        std = np.std(residuos_clase)
        iqr = np.percentile(residuos_clase, 75) - np.percentile(residuos_clase, 25)
        maximo = np.max(residuos_clase)
        minimo = np.min(residuos_clase)
        resultados_estadisticos[(p, clase)] = {
            'media': media,
            'std': std,
            'iqr': iqr,
            'max': maximo,
            'min': minimo,
            'n': len(residuos_clase)
        }

        # Análisis espectral (solo si hay suficientes datos)
        residuos_centered = residuos_clase - media
        N_points = len(residuos_centered)
        if N_points < 32:
            continue
        # FFT (sin ventana, para comparación directa)
        espectro = np.abs(fft(residuos_centered))[:N_points // 2]
        frecuencias = fftfreq(N_points, d=1)[:N_points // 2]
        resultados_frecuencias[(p, clase)] = (frecuencias, espectro)

        # Graficar espectro
        plt.figure(figsize=(7, 3))
        plt.plot(frecuencias, espectro, label=f'N mod {p} = {clase}')
        plt.xlabel('Frecuencia')
        plt.ylabel('Potencia')
        plt.title(f'Espectro residuos: N mod {p} = {clase}')
        plt.xlim(0, 0.1)
        plt.tight_layout()
        plt.savefig(f'resultados_corridas/riemann_vs_frecuencias_20250708_232216/espectro_residuos_mod{p}_{clase}.png')
        plt.close()

# Guardar estadísticas descriptivas
df_stats = pd.DataFrame.from_dict(resultados_estadisticos, orient='index')
df_stats.index = pd.MultiIndex.from_tuples(df_stats.index, names=['primo', 'clase'])
df_stats.to_csv('resultados_corridas/riemann_vs_frecuencias_20250708_232216/estadisticas_residuos_modular.csv')

print('Análisis modular y espectral completado. Ver archivos estadisticas_residuos_modular.csv y espectro_residuos_mod*_*.png')
