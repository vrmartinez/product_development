"""
Módulo de operadores para compatibilidad con notebooks.

Este módulo re-exporta los transformadores del paquete product_development
para compatibilidad hacia atrás con notebooks existentes.
"""
import sys
sys.path.insert(0, '..')

# Re-exportar desde el paquete principal
from product_development.transformers import Mapper, SimpleCategoricalImputer

__all__ = ['Mapper', 'SimpleCategoricalImputer']
