"""
Configuración de pytest para el proyecto Athena Prime.
"""

import pytest
import numpy as np

# Configuración de numpy para pruebas
np.random.seed(42)  # Para resultados reproducibles

# Configuración común para todas las pruebas
@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Configura variables de entorno para pruebas."""
    monkeypatch.setenv("PYTHONHASHSEED", "42")
