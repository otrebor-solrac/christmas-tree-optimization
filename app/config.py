"""
Configuration module for the Christmas Tree Packing algorithm.
Contains constants and configuration settings.
"""
from decimal import Decimal, getcontext

# Configurar precisión para Decimal
getcontext().prec = 25

# Factor de escala para cálculos de precisión
SCALE_FACTOR = Decimal('1e15')

