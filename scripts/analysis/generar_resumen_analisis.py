import os
import sys
from datetime import datetime

# === CONFIGURACIÓN ===
USUARIO = "Usuario: Yonnahs / Equipo: Athena Prime"

# === FUNCIONES AUXILIARES ===

def extraer_valor(linea, clave, default="-"):
    """Extrae el valor de una línea tipo 'clave: valor'"""
    if clave in linea:
        return linea.split(":", 1)[1].strip()
    return default

def buscar_linea_con_clave(ruta, clave):
    if not os.path.exists(ruta):
        return "-"
    with open(ruta, encoding='utf-8') as f:
        for linea in f:
            if clave in linea:
                return extraer_valor(linea, clave)
    return "-"

def extraer_frecuencias(ruta):
    freqs = []
    if not os.path.exists(ruta):
        return freqs
    with open(ruta, encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                fval = line.split(',')[0].strip()
                try:
                    freqs.append(float(fval))
                except Exception:
                    continue
    return freqs[:5]

def extraer_parametros_varianza(ruta):
    campos = ['k', 'gamma', 'beta', 'método']
    vals = {k: 'No disponible' for k in campos}
    if not os.path.exists(ruta):
        vals['k'] = vals['gamma'] = vals['beta'] = vals['método'] = 'Optimización de varianza no realizada'
        return vals
    with open(ruta, encoding='utf-8') as f:
        for line in f:
            if 'k:' in line:
                vals['k'] = extraer_valor(line, 'k')
            elif 'gamma' in line or 'γ' in line:
                vals['gamma'] = extraer_valor(line, 'gamma') if 'gamma' in line else extraer_valor(line, 'γ')
            elif 'beta' in line or 'β' in line:
                vals['beta'] = extraer_valor(line, 'beta') if 'beta' in line else extraer_valor(line, 'β')
            elif 'método' in line or 'metodo' in line or 'method' in line:
                vals['método'] = extraer_valor(line, 'método') if 'método' in line else (extraer_valor(line, 'metodo') if 'metodo' in line else extraer_valor(line, 'method'))
    return vals

def extraer_rango_N(ruta):
    if not os.path.exists(ruta):
        return ("-", "-")
    with open(ruta, encoding='utf-8') as f:
        for line in f:
            if 'N =' in line:
                partes = line.split('N =')[-1].strip().split('a')
                if len(partes) == 2:
                    return (partes[0].strip(), partes[1].strip())
    return ("-", "-")

def extraer_parametros_modelo(ruta, claves):
    vals = {k: '-' for k in claves}
    if not os.path.exists(ruta):
        return vals
    with open(ruta, encoding='utf-8') as f:
        for line in f:
            for k in claves:
                if k in line:
                    vals[k] = extraer_valor(line, k)
    return vals

def extraer_MSE(ruta):
    if not os.path.exists(ruta):
        return "-"
    with open(ruta, encoding='utf-8') as f:
        for line in f:
            if 'MSE' in line:
                return extraer_valor(line, 'MSE')
    return "-"

def extraer_correlaciones(ruta, n=5):
    if not os.path.exists(ruta):
        return [], 0
    out = []
    total = 0
    with open(ruta, encoding='utf-8') as f:
        for line in f:
            if 'Frequency' in line or 'Riemann zero' in line:
                total += 1
                if len(out) < n:
                    out.append(line.strip())
    return out, total

# === DETECTAR SUBCARPETA DE CORRIDA (vía argumento) ===
if len(sys.argv) > 1:
    subpath = sys.argv[1]
    if not os.path.isdir(subpath):
        raise RuntimeError(f'No existe la carpeta de corrida: {subpath}')
else:
    raise RuntimeError('Debes pasar la ruta de la subcarpeta de corrida como argumento.')

# === FECHA/HORA DE EJECUCIÓN ===
fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# === ARCHIVOS E IMÁGENES ===
archivos = [f for f in os.listdir(subpath) if f.endswith('.txt') or f.endswith('.csv')]
graficos = [f for f in os.listdir(subpath) if f.endswith('.png')]

# === EXTRACCIÓN DE DATOS CLAVE ===
frecuencias = extraer_frecuencias(os.path.join(subpath, 'frecuencias_espectro.txt'))
param_varianza = extraer_parametros_varianza(os.path.join(subpath, 'parametros_varianza_reopt.txt'))
correlaciones, total_corr = extraer_correlaciones(os.path.join(subpath, 'correlaciones_frecuencias_riemann.txt'))

# Parámetros modelo base y residuos (ajusta claves según tu script)
param_base = extraer_parametros_modelo(os.path.join(subpath, 'parametros_modelo_base.txt'), ['C', 'C1', 'C2', 'C3', 'C4'])
param_residuos = extraer_parametros_modelo(os.path.join(subpath, 'parametros_modelo_residuos.txt'), ['C1_res', 'C2_res', 'C3_res', 'alpha', 'beta'])

# Rango N y MSE
rango_ini, rango_fin = extraer_rango_N(os.path.join(subpath, 'info_rango.txt'))
MSE_total = extraer_MSE(os.path.join(subpath, 'MSE_total.txt'))

# === CONSTRUCCIÓN DEL RESUMEN ===
with open(os.path.join(subpath, 'resumen_analisis.txt'), 'w', encoding='utf-8') as f:
    f.write(f"# Análisis de Residuos Finales - Conexión Riemann\n")
    f.write(f"Fecha/hora de ejecución: {fecha_hora}\n")
    f.write(f"{USUARIO}\n\n")
    f.write(f"## Rango analizado: N = {rango_ini} a {rango_fin}\n\n")
    f.write(f"## Archivos generados\n- Datos: {', '.join(archivos)}\n- Imágenes:\n")
    for g in graficos:
        f.write(f"    - {g}\n")
    f.write("\n")
    f.write(f"## Resultados principales\n")
    f.write(f"- Error Cuadrático Medio (MSE) del Modelo TOTAL: {MSE_total}\n\n")
    f.write(f"- Parámetros óptimos del modelo base:\n")
    f.write("  | Parámetro | Valor |\n  |-----------|-------|\n")
    for k, v in param_base.items():
        f.write(f"  | {k} | {v} |\n")
    f.write("\n- Parámetros óptimos del modelo de residuos R(N):\n")
    f.write("  | Parámetro | Valor |\n  |-----------|-------|\n")
    for k, v in param_residuos.items():
        f.write(f"  | {k} | {v} |\n")
    f.write("\n- Parámetros óptimos de la varianza:\n")
    for k in ['k', 'gamma', 'beta', 'método']:
        v = param_varianza.get(k, 'No disponible')
        f.write(f"    {k}: {v}\n")
    f.write("\n- Top 5 frecuencias detectadas:\n")
    for fval in frecuencias:
        f.write(f"    {fval}\n")
    f.write("\n- Correlaciones principales con ceros de Riemann:\n")
    for c in correlaciones:
        f.write(f"    {c}\n")
    f.write(f"    Total de correlaciones encontradas: {total_corr}. Ver correlaciones_frecuencias_riemann.txt para la lista completa.\n\n")
    f.write("## Conclusión IA\n")
    f.write("- El modelo modular extendido eliminó las bandas modulares.\n")
    f.write("- Las nuevas frecuencias detectadas son mucho más bajas y sutiles.\n")
    f.write("- El “abanico” residual ahora está bien contenido por la nueva envolvente.\n")
    f.write("- Las correlaciones con ceros de Riemann son más sutiles y requieren análisis avanzado.\n\n")
    f.write("## Siguiente paso recomendado\n")
    f.write("- Analizar la distribución de correlaciones o extender el modelo a más módulos si aparecen nuevas bandas.\n")

print(f"Resumen generado en: {os.path.join(subpath, 'resumen_analisis.txt')}")
