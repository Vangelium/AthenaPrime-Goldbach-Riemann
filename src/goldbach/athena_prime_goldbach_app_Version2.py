import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expi
from scipy.signal import lombscargle
from functools import lru_cache

def sieve_of_eratosthenes(limit):
    is_prime = np.full(limit + 1, True)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[p]:
            is_prime[p*p:limit+1:p] = False
    primes = np.where(is_prime)[0]
    return primes, is_prime

def prime_factorize(n):
    factors = {}
    num = n
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def calculate_singular_series(factors):
    product = 1.0
    for p in factors:
        if p > 2:
            product *= (p - 1) / (p - 2)
    return product

@lru_cache(maxsize=None)
def li2(x):
    if x < 2: return 0
    log_x = np.log(x)
    return expi(log_x) - (x / log_x)

def constrained_residual_model(X, C1_res, C2_res, C3_res, alpha, K):
    N, omega_N = X
    beta = K - alpha
    epsilon = 1e-9
    term1 = C1_res * (N**alpha) * np.log(omega_N + 1)
    term2 = C2_res * np.sqrt(N) * ((omega_N + epsilon)**beta)
    term3 = C3_res
    return term1 + term2 + term3

# Variante 1: Solo término C1 * N^alpha * log(omega_N+1)
def residual_model_solo_C1(X, C1_res, C3_res, alpha):
    N, omega_N = X
    epsilon = 1e-9
    return C1_res * (N**alpha) * np.log(omega_N + 1) + C3_res

# Variante 2: Solo término C2 * sqrt(N) * (omega_N+epsilon)^beta
def residual_model_solo_C2(X, C2_res, C3_res, beta):
    N, omega_N = X
    epsilon = 1e-9
    return C2_res * np.sqrt(N) * ((omega_N + epsilon)**beta) + C3_res

def refined_final_residual_model(X, a, gamma,
                                 c0_3, c1_3, c2_3,
                                 c0_5, c1_5, c2_5, c3_5, c4_5,
                                 c0_7, c1_7, c2_7, c3_7, c4_7, c5_7, c6_7):
    N, omega_N = X
    epsilon = 1e-9
    mod3 = N % 3
    mod5 = N % 5
    mod7 = N % 7
    mod_term_3 = (mod3 == 0) * c0_3 + (mod3 == 1) * c1_3 + (mod3 == 2) * c2_3
    mod_term_5 = (mod5 == 0) * c0_5 + (mod5 == 1) * c1_5 + (mod5 == 2) * c2_5 + (mod5 == 3) * c3_5 + (mod5 == 4) * c4_5
    mod_term_7 = (mod7 == 0) * c0_7 + (mod7 == 1) * c1_7 + (mod7 == 2) * c2_7 + (mod7 == 3) * c3_7 + (mod7 == 4) * c4_7 + (mod7 == 5) * c5_7 + (mod7 == 6) * c6_7
    term1 = a * (N**gamma) * np.log(omega_N + epsilon)
    return term1 + mod_term_3 + mod_term_5 + mod_term_7

def run_full_analysis_for_range(start_n, end_n, base_constants, K_constraint, corrida_dir):
    C, C1, C2, C3, C4 = base_constants
    print(f"\nANÁLISIS: N = {start_n} a {end_n}")
    primes, is_prime_map = sieve_of_eratosthenes(end_n)
    results = []
    step = 2 if (end_n - start_n) < 50000 else 20
    for n in range(start_n, end_n + 1, step):
        count = 0
        for p in primes:
            if p > n / 2: break
            if n - p < len(is_prime_map) and is_prime_map[n - p]: count += 1
        gN = count
        theoretical_term = li2(n)
        deltaN = gN - theoretical_term
        factors = prime_factorize(n)
        omegaN = len(factors)
        singular_series = calculate_singular_series(factors)
        base_prediction = (C * singular_series - 1) * theoretical_term
        correction_term1 = (C1 * omegaN) + (C2 * np.sqrt(n) * omegaN) + (C3 * np.sqrt(n))
        correction_term2 = C4 * omegaN * np.log(n)
        deltaPredictedN = base_prediction + correction_term1 + correction_term2
        results.append({
            'N': n, 'g(N)': gN, 'Δ Real': deltaN, 'Δ Predicho (Base)': deltaPredictedN, 'ω(N)': omegaN
        })
    df = pd.DataFrame(results)
    df['Residuo (Base)'] = df['Δ Real'] - df['Δ Predicho (Base)']

    # Ajuste del primer residuo R(N)
    x_data_r1 = (df['N'].values, df['ω(N)'].values)
    y_data_r1 = df['Residuo (Base)'].values
    try:
        model_to_fit_r1 = lambda X, C1, C2, C3, alpha: constrained_residual_model(X, C1, C2, C3, alpha, K=K_constraint)
        popt_r1, _ = curve_fit(model_to_fit_r1, x_data_r1, y_data_r1, p0=[1, 1, 1, 0.5], maxfev=20000)
        df['R Predicho'] = constrained_residual_model(x_data_r1, *popt_r1, K=K_constraint)
        df['Δ Predicho (Mejorado)'] = df['Δ Predicho (Base)'] + df['R Predicho']
        df['Residuo (Final)'] = df['Δ Real'] - df['Δ Predicho (Mejorado)']

        # Ajuste del residuo final R_final(N) con términos modulares extendidos
        x_data_r2 = (df['N'].values, df['ω(N)'].values)
        y_data_r2 = df['Residuo (Final)'].values
        # p0: [a, gamma, c0_3, c1_3, c2_3, c0_5, ..., c4_5, c0_7, ..., c6_7]
        p0 = [0, 0.5] + [0]*3 + [0]*5 + [0]*7
        popt_r2, _ = curve_fit(refined_final_residual_model, x_data_r2, y_data_r2, p0=p0, maxfev=50000)
        df['R_final Predicho'] = refined_final_residual_model(x_data_r2, *popt_r2)
        df['Δ Predicho (Total)'] = df['Δ Predicho (Mejorado)'] + df['R_final Predicho']

        # --- Modelos simplificados de residuos ---
        # Modelo simplificado: Solo C1*N^alpha*log(omega_N+1) + C3
        def model1(X, C1, C3, alpha):
            return residual_model_solo_C1(X, C1, C3, alpha)
        popt_model1, _ = curve_fit(model1, x_data_r1, y_data_r1, p0=[1, 1, 0.5], maxfev=20000)
        resid_model1 = residual_model_solo_C1(x_data_r1, *popt_model1)
        mse_model1 = np.mean((y_data_r1 - resid_model1)**2)

        # Guardar los parámetros y el MSE en la subcarpeta de la corrida
        with open(os.path.join(corrida_dir, 'parametros_modelo_residuos_soloC1.txt'), 'w', encoding='utf-8') as f:
            f.write(f'C1_res: {popt_model1[0]}\n')
            f.write(f'C3_res: {popt_model1[1]}\n')
            f.write(f'alpha: {popt_model1[2]}\n')
            f.write(f'MSE: {mse_model1}\n')

        print("\n--- Modelo simplificado de R(N): Solo C1*N^alpha*log(omega_N+1) + C3 ---")
        print(f"Parámetros: {popt_model1}, MSE: {mse_model1:.4e}")
        print("--------------------------------------------------------------\n")

        return df, popt_r1, popt_r2, (popt_model1, mse_model1), None

    except RuntimeError as e:
        print(f"Optimización fallida para este rango. Error: {e}")
        return None, None, None, None, None

def spectral_and_modular_analysis(df, output_dir):
    import os
    print("\nAnálisis Espectral y Modular...")
    os.makedirs(output_dir, exist_ok=True)
    t = df['N'].values
    y = df['Residuo (Total)'].values
    freqs = np.linspace(0.0001, 0.1, 10000)
    power = lombscargle(t, y, freqs, normalize=True)
    plt.figure(figsize=(10,5))
    plt.plot(freqs, power, color='navy')
    plt.title('Espectro de Residuos Finales')
    plt.xlabel('Frecuencia')
    plt.ylabel('Potencia')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "espectro.png"))
    plt.close()

    # Encontrar las 5 frecuencias más prominentes
    top_idxs = np.argsort(power)[-5:][::-1]
    top_freqs = freqs[top_idxs]
    top_powers = power[top_idxs]
    print("Top 5 frecuencias prominentes:")
    for f, p in zip(top_freqs, top_powers):
        print(f"Frecuencia: {f:.5f}, Potencia: {p:.5f}")

    # Guardar en archivo
    with open(os.path.join(output_dir, 'frecuencias_espectro.txt'), 'w') as f:
        f.write("Top 5 frecuencias prominentes (frecuencia, potencia):\n")
        for freq, pow_ in zip(top_freqs, top_powers):
            f.write(f"{freq:.8f}, {pow_:.8f}\n")

    for p in [3, 5, 7, 11]:
        plt.figure()
        for m in range(p):
            plt.scatter(df[df[f'N_mod_{p}']==m]['N'], df[df[f'N_mod_{p}']==m]['Residuo (Total)'], s=8, label=f'N mod {p}={m}')
        plt.title(f'Residuos por clase N mod {p}')
        plt.xlabel('N')
        plt.ylabel('Residuo (Total)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mod_{p}.png"))
        plt.close()
    return top_freqs

if __name__ == "__main__":
    # --- Configuración ---
    BASE_CONSTANTS = [0.4074767, 1.3017311, 0.08961985, 0.8145583, 2.0090410]
    K_CONSTRAINT = 1.13
    RANGES_TO_ANALYZE = [
        (250000, 300000),
        (50000, 100000)
    ]

    import os
    from datetime import datetime
    # Crear carpeta raíz si no existe
    root_corridas = os.path.join(os.getcwd(), 'resultados_corridas')
    os.makedirs(root_corridas, exist_ok=True)
    # Crear subcarpeta única para esta corrida
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    corrida_dir = os.path.join(root_corridas, f'riemann_vs_frecuencias_{timestamp}')
    os.makedirs(corrida_dir, exist_ok=True)

    full_results_df = pd.DataFrame()
    for r, (start_range, end_range) in enumerate(RANGES_TO_ANALYZE):
        result = run_full_analysis_for_range(start_range, end_range, BASE_CONSTANTS, K_CONSTRAINT, corrida_dir)
        if result is not None:
            result_df, popt_r1, popt_r2, (popt_model1, mse_model1), _ = result
            full_results_df = pd.concat([full_results_df, result_df], ignore_index=True)
            # Guardar parámetros y MSE de modelos simplificados en archivos auxiliares
            with open(os.path.join(corrida_dir, f'parametros_modelo_residuos_soloC1.txt'), 'w', encoding='utf-8') as f:
                f.write(f'C1_res: {popt_model1[0]}\nC3_res: {popt_model1[1]}\nalpha: {popt_model1[2]}\nMSE: {mse_model1}\n')

        else:
            print(f"Rango {start_range}-{end_range} omitido por error.")
    if not full_results_df.empty:
        full_results_df['Residuo (Total)'] = full_results_df['Δ Real'] - full_results_df['Δ Predicho (Total)']
        for p in [3, 5, 7, 11]:
            full_results_df[f'N_mod_{p}'] = full_results_df['N'] % p
        print(f"Guardando resultados y gráficas en: {corrida_dir}")
        prominent_freqs = spectral_and_modular_analysis(full_results_df, output_dir=corrida_dir)
        full_results_df.to_csv(os.path.join(corrida_dir, "goldbach_residuos.csv"), index=False)
        print("Análisis completado. Consulta los archivos y gráficos generados para interpretar los resultados.")

        # === Post-procesado automático: varianza, correlaciones y resumen ===
        import subprocess
        try:
            print("Ejecutando optimización de varianza reopt...")
            subprocess.run(['python', 'reopt_varianza_residuos.py', corrida_dir], check=True)
        except Exception as e:
            print(f"Error al ejecutar reopt_varianza_residuos.py: {e}")
        try:
            print("Ejecutando comparación de frecuencias de Riemann...")
            subprocess.run(['python', 'comparacion_frecuencias_riemann.py', corrida_dir], check=True)
        except Exception as e:
            print(f"Error al ejecutar comparacion_frecuencias_riemann.py: {e}")
        try:
            print("Generando resumen autocontenible...")
            subprocess.run(['python', 'generar_resumen_analisis.py', corrida_dir], check=True)
        except Exception as e:
            print(f"Error al ejecutar generar_resumen_analisis.py: {e}")

        # Guardar archivos auxiliares para el resumen
        with open(os.path.join(corrida_dir, 'info_rango.txt'), 'w', encoding='utf-8') as f:
            f.write(f'Rango analizado: N = {RANGES_TO_ANALYZE[0][0]} a {RANGES_TO_ANALYZE[-1][1]}\n')
        if 'Residuo (Total)' in full_results_df.columns:
            mse_total = np.mean(full_results_df['Residuo (Total)']**2)
            with open(os.path.join(corrida_dir, 'MSE_total.txt'), 'w', encoding='utf-8') as f:
                f.write(f'MSE: {mse_total:.8e}\n')
        if 'BASE_CONSTANTS' in globals():
            popt_base = BASE_CONSTANTS
            with open(os.path.join(corrida_dir, 'parametros_modelo_base.txt'), 'w', encoding='utf-8') as f:
                f.write(f'C: {popt_base[0]}\nC1: {popt_base[1]}\nC2: {popt_base[2]}\nC3: {popt_base[3]}\nC4: {popt_base[4]}\n')
        if 'popt_model1' in locals() and popt_model1 is not None:
            with open(os.path.join(corrida_dir, 'parametros_modelo_residuos.txt'), 'w', encoding='utf-8') as f:
                f.write(f'C1_res: {popt_model1[0]}\nC3_res: {popt_model1[1]}\nalpha: {popt_model1[2]}\nMSE: {mse_model1}\n')

        # Generar resumen automático del análisis, pasando la ruta de la corrida
        import subprocess
        try:
            subprocess.run(['python', 'generar_resumen_analisis.py', corrida_dir], check=True)
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo generar el resumen automático: {e}")
        print(f"¡Listo! Consulta todos los archivos en '{corrida_dir}' para interpretar los patrones.")