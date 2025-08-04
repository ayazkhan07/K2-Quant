"""
K2 Quant Data Layer Package

Provides database management and data access functionality.
"""

from .db_manager import db_manager, DatabaseManager

__all__ = ['db_manager', 'DatabaseManager']