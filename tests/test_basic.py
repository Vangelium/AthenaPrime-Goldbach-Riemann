"""
Test básicos para el proyecto Athena Prime.
"""

def test_imports():
    """Verifica que se puedan importar los módulos principales."""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import scipy
        import sympy
        import mpmath
        import joblib
        assert True
    except ImportError as e:
        assert False, f"Error de importación: {e}"

def test_basic_math():
    """Prueba operaciones matemáticas básicas."""
    import numpy as np
    assert 1 + 1 == 2
    assert np.array([1, 2, 3]).sum() == 6
