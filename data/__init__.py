"""
Módulo de carregamento e download de dados
"""
from .loader import DataLoader
from .downloader import KaggleDownloader

__all__ = ['DataLoader', 'KaggleDownloader']

