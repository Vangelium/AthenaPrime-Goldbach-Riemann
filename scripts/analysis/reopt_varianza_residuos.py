import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

import sys

# --- Directorio de resultados: se pasa como argumento ---
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
    if not os.path.isdir(output_dir):
        raise RuntimeError(f'No existe la carpeta de corrida: {output_dir}')
else:
    raise RuntimeError('Debes pasar la ruta de la subcarpeta de corrida como argumento.')

# --- 1. Cargar los Residuos Finales ---
try:
    df_residuos = pd.read_csv(os.path.join(output_dir, 'goldbach_residuos.csv'))
    print("Residuos finales cargados correctamente.")
except FileNotFoundError:
    print(f"Error: No se encontró '{os.path.join(output_dir, 'goldbach_residuos.csv')}'. Asegúrate de que las fases anteriores se ejecutaron y guardaron los datos.")
    exit()

# Asegurarse de que no haya ceros o negativos en omega_N para el logaritmo
epsilon = 1e-9

# --- 2. Definir el Modelo de Varianza (Función para curve_fit) ---
def variance_model_log(X_data, log_k, gamma, beta):
    N, omega_N = X_data
    return log_k + gamma * np.log(N) + beta * np.log(omega_N + epsilon)

# Preparar los datos para la optimización
y_data_variance = np.log(np.abs(df_residuos['Residuo (Total)'].values) + epsilon)
x_data_variance = (df_residuos['N'].values, df_residuos['ω(N)'].values)

# --- 3. Realizar Regresión No Lineal (Optimización) ---
print("\nIniciando optimización de la envolvente de la varianza...")

best_popt_variance = None
min_mse_variance = np.inf

initial_guesses_variance = [
    [np.log(0.001), 0.5, 2.0],
    [np.log(0.0005), 0.4, 2.5],
    [np.log(0.005), 0.6, 3.5]
]

from scipy.optimize import least_squares

def residuals_varianza(params, X, y):
    log_k, gamma, beta = params
    pred = variance_model_log(X, log_k, gamma, beta)
    return y - pred

best_result = None
min_cost = np.inf
for p0_var in initial_guesses_variance:
    try:
        res = least_squares(residuals_varianza, p0_var, args=(x_data_variance, y_data_variance), loss='soft_l1', max_nfev=100000)
        if res.cost < min_cost:
            min_cost = res.cost
            best_result = res
    except Exception as e:
        pass

if best_result is None:
    print("Error: No se pudo optimizar el modelo de varianza (regresión robusta).")
    exit()

log_k_opt, gamma_opt, beta_opt = best_result.x
k_opt = np.exp(log_k_opt)

# Guardar los parámetros óptimos en un archivo TXT (solo ASCII)
with open(os.path.join(output_dir, 'parametros_varianza_reopt.txt'), 'w', encoding='utf-8') as f:
    f.write('Parametros optimos para la ENVOLVENTE DEL ERROR V(N, w(N)) (re-optimizado, regresion robusta soft_l1):\n')
    f.write(f'  k: {k_opt:.8f}\n')
    f.write(f'  gamma: {gamma_opt:.8f}\n')
    f.write(f'  beta: {beta_opt:.8f}\n')
    f.write('  metodo: least_squares (soft_l1)\n')
print('parametros_varianza_reopt.txt guardado correctamente (ASCII)')

print("\nOptimización de la envolvente completada con éxito.")
print("Parámetros óptimos para la ENVOLVENTE DEL ERROR V(N, ω(N)):")
print(f"  k: {k_opt:.6f}")
print(f"  gamma (γ): {gamma_opt:.6f}")
print(f"  beta (β) de varianza: {beta_opt:.6f}")

# --- 4. Calcular la Envolvente Predicha ---
def predicted_variance_envelope(N, omega_N, k, gamma, beta):
    return k * (N**gamma) * ((omega_N + epsilon)**beta)

df_residuos['V_predicha'] = predicted_variance_envelope(df_residuos['N'].values, df_residuos['ω(N)'].values, k_opt, gamma_opt, beta_opt)

# --- 5. Visualizar Resultados ---
plt.figure(figsize=(12, 7))
plt.scatter(df_residuos['N'], df_residuos['Residuo Final Total'], s=5, alpha=0.5, label='Residuo Final', color='cyan')
plt.plot(df_residuos['N'], df_residuos['V_predicha'], color='lime', linestyle='--', label='Envolvente de Error +V(N,ω(N))')
plt.plot(df_residuos['N'], -df_residuos['V_predicha'], color='lime', linestyle='--', label='Envolvente de Error -V(N,ω(N))')
plt.axhline(0, color='red', linestyle='-', linewidth=1)
plt.xlabel('N')
plt.ylabel('Residuo Final Total')
plt.title('Residuos del Modelo TOTAL y su Envolvente de Incertidumbre Predicha (Re-optimizado)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residuos_finales_con_envolvente_reopt.png'))
plt.close()

print(f"\nGráfico de residuos con envolvente re-optimizada guardado en '{os.path.join(output_dir, 'residuos_finales_con_envolvente_reopt.png')}'")

# Imprimir en consola SOLO ASCII para máxima compatibilidad
print("\n==============================")
print("Optimal parameters for variance envelope V(N, w(N)) (re-optimized):")
print(f"  k: {k_opt:.8f}")
print(f"  gamma: {gamma_opt:.8f}")
print(f"  beta: {beta_opt:.8f}")
print("==============================\n")

print("Variance optimization complete. Check the plot and parameters for the new envelope.")
