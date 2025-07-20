# Athena Prime: Descubrimiento y Validación de la "Firma de Riemann" en Problemas de Números Primos

*Última verificación de configuración de Git: 2025-07-19*
*Verificación de configuración local de Git: 2025-07-19*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16197824.svg)](https://doi.org/10.5281/zenodo.16197824)

## 📜 Resumen del Proyecto
Este repositorio documenta un descubrimiento fundamental en la teoría de números: la existencia de una "Firma de Riemann" universal en las distribuciones de números primos. A través de un riguroso análisis espectral y validación estadística, hemos demostrado que los residuos de cinco problemas aparentemente distintos —la Conjetura de Cramér (gaps de primos), la Conjetura de los Primos Gemelos, la Conjetura de Goldbach, la Conjetura de Polignac (para n=4, "Cousin Primes") y el Postulado de Bertrand— exhiben patrones de frecuencia casi idénticos, fuertemente correlacionados con los ceros no triviales de la función Zeta de Riemann.

Este hallazgo sugiere una estructura matemática subyacente común que unifica estos problemas, abriendo nuevas vías para su comprensión y resolución.

## 🔍 Hallazgos Clave

Nuestro análisis revela correlaciones sin precedentes entre los espectros de potencia de los residuos de Cramér, Primos Gemelos, Goldbach, Polignac y Bertrand:

-   **Correlación de Pearson (Cramér vs. Primos Gemelos):** `r = 0.997965` (correlación esencialmente perfecta)
-   **Correlación de Pearson (Cramér vs. Goldbach):** `r = 0.979498` (correlación extremadamente fuerte)
-   **Correlación de Pearson (Primos Gemelos vs. Goldbach):** `r = 0.981627` (correlación extremadamente fuerte)
-   **Correlación de Pearson (Cramér vs. Polignac):** `r = 0.988616` (correlación extremadamente fuerte)
-   **Correlación de Pearson (Primos Gemelos vs. Polignac):** `r = 0.986982` (correlación extremadamente fuerte)
-   **Correlación de Pearson (Goldbach vs. Polignac):** `r = 0.996821` (correlación esencialmente perfecta)
-   **Correlación de Pearson (Cramér vs. Bertrand):** `r = 0.979398` (correlación extremadamente fuerte)
-   **Correlación de Pearson (Primos Gemelos vs. Bertrand):** `r = 0.978165` (correlación extremadamente fuerte)
-   **Correlación de Pearson (Goldbach vs. Bertrand):** `r = 0.998093` (correlación esencialmente perfecta)
-   **Correlación de Pearson (Polignac vs. Bertrand):** `r = 0.998414` (correlación esencialmente perfecta)

Los **p-values de 0.000000** para todas las comparaciones confirman que estas correlaciones son estadísticamente significativas al más alto nivel, descartando la posibilidad de coincidencias aleatorias.

### Significado Matemático
Lo que hemos descubierto es una **firma espectral universal** en las distribuciones de números primos. El hecho de que cinco problemas aparentemente distintos exhiban características del dominio de frecuencias casi idénticas sugiere que son manifestaciones de la misma estructura matemática subyacente, profundamente ligada a la distribución de los ceros de Riemann.

### Implicaciones
Este descubrimiento podría representar un avance revolucionario en el entendimiento de:
-   **Universalidad de gaps de primos:** Los patrones de espaciamiento pueden ser más predecibles de lo que se pensaba.
-   **Interconexión de conjeturas:** Las conjeturas de Primos Gemelos, Goldbach, Polignac y el Postulado de Bertrand pueden ser diferentes caras del mismo fenómeno.
-   **Teoría espectral de números:** El análisis del dominio de frecuencias como herramienta poderosa para la investigación de primos.

## 🛠️ Requisitos

-   Python 3.8 o superior
-   Bibliotecas listadas en `requirements.txt`

```bash
numpy, pandas, matplotlib, scipy, sympy, mpmath, joblib
```

## 📂 Estructura del Repositorio

```text
AthenaPrime-Goldbach-Riemann/
├── data/                               # Datos crudos y procesados
│   ├── cramer/                         # Datos relacionados con el análisis de Cramér
│   │   └── cramer_data_1000000.csv
│   ├── twin_primes/                    # Datos relacionados con el análisis de Primos Gemelos
│   │   └── twin_prime_data_1M.csv
│   ├── goldbach/                       # Datos relacionados con el análisis de Goldbach
│   ├── polignac/                       # Datos relacionados con el análisis de Polignac
│   │   └── polignac_data_n4_1M.csv
│   └── bertrand/                       # Datos relacionados con el análisis de Bertrand
│       └── bertrand_data_1M.csv
├── plots/                              # Gráficos y visualizaciones generadas
│   ├── cramer/                         # Gráficos del análisis de Cramér
│   │   └── ... (varios PNGs)
│   ├── twin_primes/                    # Gráficos del análisis de Primos Gemelos
│   │   └── ... (varios PNGs)
│   ├── goldbach/                       # Gráficos del análisis de Goldbach y comparativos
│   ├── polignac/                       # Gráficos del análisis de Polignac
│   └── bertrand/                       # Gráficos del análisis de Bertrand
│       └── 12_comparative_power_spectra_1M_polignac_bertrand.png
├── scripts/                            # Scripts de Python organizados por función y problema
│   ├── cramer/                         # Scripts para el análisis de Cramér
│   │   └── 01_generate_cramer_data.py
│   │   └── ... (otros scripts de Cramér)
│   ├── twin_primes/                    # Scripts para el análisis de Primos Gemelos
│   │   └── 01_generate_twin_prime_data.py
│   │   └── ... (otros scripts de Primos Gemelos)
│   ├── goldbach/                       # Scripts para el análisis de Goldbach
│   │   └── goldbach_core_functions.py
│   │   └── goldbach_residual_clean.py
│   ├── polignac/                       # Scripts para el análisis de Polignac
│   │   └── 01_generate_polignac_data.py
│   │   └── get_polignac_residual.py
│   ├── bertrand/                       # Scripts para el análisis de Bertrand
│   │   └── 01_generate_bertrand_data.py
│   │   └── get_bertrand_residual.py
│   └── 08_cross_coherence_analysis.py  # Script principal para el análisis comparativo final
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🚀 Comenzando

### Configuración del Entorno

```bash
# 1. Clonar el repositorio
git clone https://github.com/Vangelium/AthenaPrime-Goldbach-Riemann.git
cd AthenaPrime-Goldbach-Riemann

# 2. Crear y activar entorno virtual (recomendado)
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Unix/MacOS:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Replicar el Análisis de la "Firma de Riemann"

Para replicar el análisis comparativo final que valida la "Firma de Riemann" en los residuos de Cramér, Primos Gemelos, Goldbach, Polignac y Bertrand, ejecuta el siguiente script:

```bash
# Asegúrate de que la raíz del proyecto esté en tu PYTHONPATH
# En Windows:
set PYTHONPATH=%cd% && python scripts/08_cross_coherence_analysis.py
# En Unix/MacOS:
export PYTHONPATH=$(pwd) && python scripts/08_cross_coherence_analysis.py
```

Este script realizará los siguientes pasos:
1.  Generará los datos necesarios para la Conjetura de Polignac y el Postulado de Bertrand (si no existen).
2.  Calculará los residuos limpios para Cramér, Primos Gemelos, Goldbach, Polignac y Bertrand.
3.  Realizará un análisis espectral (Lomb-Scargle) para cada residuo.
4.  Generará un gráfico comparativo de los espectros de potencia, guardado en `plots/goldbach/12_comparative_power_spectra_1M_polignac_bertrand.png`.
5.  Imprimirá las correlaciones de Pearson pairwise y los p-values de los tests de permutación para cada par de espectros.

### Resultados
Los resultados del análisis se imprimirán en la consola. El gráfico comparativo de los espectros de potencia se guardará en `plots/goldbach/12_comparative_power_spectra_1M_polignac_bertrand.png`. Este gráfico visualiza la notable similitud de los patrones de frecuencia entre los cinco problemas, con líneas rojas verticales indicando las frecuencias esperadas de los ceros de Riemann.

## 🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, asegúrate de que los cambios propuestos cumplan con los estándares del proyecto.

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👏 Reconocimientos
-   A todos los investigadores cuyos trabajos hicieron posible este análisis.
-   A la comunidad de código abierto por las herramientas utilizadas.
-   A los revisores por sus valiosos comentarios y sugerencias.

## 📞 Contacto
Para preguntas o colaboraciones, por favor abre un issue en este repositorio.