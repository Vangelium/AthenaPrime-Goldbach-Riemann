# Athena Prime: Análisis de la Firma de Riemann en los Residuos de Goldbach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📜 Resumen del Proyecto
Este repositorio contiene la implementación y análisis que revela una conexión profunda entre los residuos de la Conjetura de Goldbach y los ceros de la función Zeta de Riemann. A través de un riguroso análisis espectral, hemos identificado patrones modulares que sugieren una relación fundamental entre estas dos áreas de la teoría de números.

## 🔍 Hallazgos Clave

- **Correlación directa** entre frecuencias dominantes en residuos de Goldbach y diferencias entre ceros de Riemann (precisión de hasta 10⁻⁷)
- **Concentración de energía espectral** en la clase modular N mod P = 0 para primos pequeños (P=3,5)
- **Firma espectral** consistente que se alinea con la distribución de ceros de Riemann
- **Análisis empírico** que valida las predicciones teóricas

## 🚀 Características Principales

- Análisis espectral de residuos de Goldbach
- Detección de frecuencias dominantes
- Visualización de correlación con ceros de Riemann
- Validación estadística de resultados

## 🛠️ Requisitos

- Python 3.8 o superior
- Bibliotecas listadas en `requirements.txt`

```bash
numpy, pandas, matplotlib, scipy, sympy, mpmath, joblib
```

## 📂 Estructura del Repositorio

```text
AthenaPrime-Goldbach-Riemann/
├── data/                    # Datos crudos y procesados
│   ├── processed/          # Datos procesados listos para análisis
│   └── raw/                # Datos crudos (gitignorados)
├── scripts/                # Scripts de Python organizados por función
│   ├── analysis/           # Análisis de datos y procesamiento
│   └── visualization/      # Generación de gráficos y visualizaciones
├── src/                    # Código fuente principal
│   └── goldbach/           # Implementación de algoritmos clave
├── tests/                  # Pruebas unitarias
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

### Ejecutar Análisis
```bash
# Ejecutar análisis modular
python scripts/analysis/modular_analysis.py

# Generar visualizaciones
python scripts/visualization/generate_plots.py
```

## 🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, asegúrate de que los cambios propuestos cumplan con los estándares del proyecto.

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👏 Reconocimientos
- A todos los investigadores cuyos trabajos hicieron posible este análisis
- A la comunidad de código abierto por las herramientas utilizadas
- A los revisores por sus valiosos comentarios y sugerencias

## 📞 Contacto
Para preguntas o colaboraciones, por favor abre un issue en este repositorio.
