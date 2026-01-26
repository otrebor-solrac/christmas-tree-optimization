"""
Global configuration for the application.
"""
from decimal import Decimal, getcontext

# Set global precision
getcontext().prec = 25

# Scale factor used for coordinate normalization
SCALE_FACTOR = Decimal('1e15')
