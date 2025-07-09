import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURACIÓN ---
# Uso: python comparacion_frecuencias_riemann.py <ruta_subcarpeta_corrida>
if len(sys.argv) > 1:
    corrida_dir = sys.argv[1]
    if not os.path.isdir(corrida_dir):
        raise RuntimeError(f'No existe la carpeta de corrida: {corrida_dir}')
else:
    raise RuntimeError('Debes pasar la ruta de la subcarpeta de corrida como argumento.')

# Archivos de ceros de Riemann
zeros_dir = os.path.join(os.getcwd(), 'zeta_tables', 'zeros')
zero_files_to_parse = [
    os.path.join(zeros_dir, f"zeros{i}.txt") for i in range(1, 7)
]
# Leer frecuencias espectrales detectadas desde el archivo generado por el análisis
freqs_path = os.path.join(corrida_dir, 'frecuencias_espectro.txt')
detected_frequencies = []
try:
    with open(freqs_path, 'r') as f:
        for line in f:
            if ',' in line:
                try:
                    freq = float(line.strip().split(',')[0])
                    detected_frequencies.append(freq)
                except Exception:
                    continue
except FileNotFoundError:
    print(f"No se encontró el archivo de frecuencias: {freqs_path}")

# --- PARSING DE CEROS DE RIEMANN ---
all_imaginary_parts = []
for file_name in zero_files_to_parse:
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        imaginary_part = float(line)
                        all_imaginary_parts.append(imaginary_part)
                    except ValueError:
                        parts = line.split()
                        for part in parts:
                            try:
                                val = float(part)
                                if val > 1.0:
                                    all_imaginary_parts.append(val)
                                    break
                            except ValueError:
                                continue
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")
# También intentamos con "zeros of (s).txt" si existe
try:
    with open("zeros of (s).txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        imaginary_part = float(parts[1])
                        if imaginary_part > 1.0:
                            all_imaginary_parts.append(imaginary_part)
                    except ValueError:
                        continue
                elif len(parts) == 1:
                    try:
                        imaginary_part = float(parts[0])
                        if imaginary_part > 1.0:
                            all_imaginary_parts.append(imaginary_part)
                    except ValueError:
                        continue
except FileNotFoundError:
    pass
except Exception as e:
    print(f"An error occurred while reading zeros of (s).txt: {e}")

# --- FILTRADO DE CEROS ---
riemann_zeros_df = pd.DataFrame(sorted(all_imaginary_parts), columns=['ImaginaryPart'])
# Filtro de plausibilidad: >10 y <1_000_000
riemann_zeros_df = riemann_zeros_df[(riemann_zeros_df['ImaginaryPart'] > 10) & (riemann_zeros_df['ImaginaryPart'] < 1_000_000)]
riemann_zeros_df = riemann_zeros_df.reset_index(drop=True)

# --- DIFERENCIAS ENTRE CEROS ---
riemann_zeros_df['Difference'] = riemann_zeros_df['ImaginaryPart'].diff()
differences = riemann_zeros_df['Difference'].dropna().values

# --- BÚSQUEDA DE CORRELACIONES ---
tolerance = 0.0001
found_correlations = []
for freq in detected_frequencies:
    for diff in differences:
        if abs(freq - diff) < tolerance:
            found_correlations.append(f"Frequency {freq} is directly close to a Riemann zero difference of {diff}.")
        if abs(diff - (2 * np.pi * freq)) < tolerance:
            found_correlations.append(f"Frequency {freq} * 2*pi ({2 * np.pi * freq}) is directly close to a Riemann zero difference of {diff}.")
        if diff != 0:
            if abs(freq / diff - 1) < tolerance:
                pass
            elif abs(freq / diff - round(freq / diff)) < tolerance and round(freq / diff) != 0:
                found_correlations.append(f"Frequency {freq} is approx {round(freq / diff)} * Riemann zero difference of {diff}.")
            elif abs(diff / freq - round(diff / freq)) < tolerance and round(diff / freq) != 0:
                found_correlations.append(f"Riemann zero difference of {diff} is approx {round(diff / freq)} * Frequency {freq}.")
            elif abs(freq - (1/diff)) < tolerance and diff != 0:
                found_correlations.append(f"Frequency {freq} is directly close to 1 / (Riemann zero difference of {diff}).")
            elif abs(freq - (diff / (2 * np.pi))) < tolerance:
                found_correlations.append(f"Frequency {freq} is directly close to (Riemann zero difference of {diff}) / (2*pi).")

# --- GUARDAR CORRELACIONES ---
correlaciones_path = os.path.join(corrida_dir, 'correlaciones_frecuencias_riemann.txt')
with open(correlaciones_path, 'w', encoding='utf-8') as f:
    if found_correlations:
        f.write("Correlaciones encontradas entre frecuencias espectrales y diferencias de ceros de Riemann:\n\n")
        for corr in found_correlations:
            f.write(corr + "\n")
    else:
        f.write("No se encontraron correlaciones directas dentro de la tolerancia especificada.\n")
print(f"Correlaciones guardadas en: {correlaciones_path}")

# --- GRÁFICO DE COMPARACIÓN ---
plt.figure(figsize=(12,6))
# Histograma de diferencias
plt.hist(differences, bins=200, color='skyblue', alpha=0.7, label='Diferencias entre ceros')
# Líneas verticales para frecuencias detectadas
for freq in detected_frequencies:
    plt.axvline(freq, color='red', linestyle='--', alpha=0.8, label=f'Frecuencia detectada: {freq}')
    plt.axvline(2*np.pi*freq, color='orange', linestyle=':', alpha=0.7, label=f'{freq} × 2π')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Comparación: Diferencias entre ceros de Riemann vs Frecuencias espectrales detectadas')
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
grafico_path = os.path.join(corrida_dir, 'comparacion_frecuencias_riemann.png')
plt.savefig(grafico_path, dpi=150)
plt.close()
print(f"Gráfico guardado en: {grafico_path}")

# --- INFO FINAL ---
print("\nResumen:")
print(f"Total de ceros analizados: {len(riemann_zeros_df)}")
print(f"Total de correlaciones encontradas: {len(found_correlations)}")
print(f"Gráfico generado: {grafico_path}")
print(f"Correlaciones guardadas: {correlaciones_path}")
