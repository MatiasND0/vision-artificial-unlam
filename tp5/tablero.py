"""Geometría medida del tablero. El ejemplo no es una medición física."""
import json
from pathlib import Path

import numpy as np

from .configuracion import (CENTROS_TABLERO_MM, DICCIONARIO, LADO_TABLERO_MM,
                            TAMANO_TABLERO_MM)
from .modelos import Tablero


def esquinas_cuadrado(lado_mm):
    """Orden canónico ArUco / IPPE_SQUARE, normal local hacia +Z."""
    if not np.isfinite(lado_mm) or lado_mm <= 0:
        raise ValueError("Lado del marcador inválido (mm)")
    h = lado_mm / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def crear_tablero_ejemplo():
    local = esquinas_cuadrado(LADO_TABLERO_MM)
    return Tablero(DICCIONARIO,
                   {i: local + [x, y, 0] for i, (x, y) in CENTROS_TABLERO_MM.items()},
                   TAMANO_TABLERO_MM, es_ejemplo=True)


def guardar_tablero(tablero, ruta):
    datos = {"version": 1, "unidades": "mm", "diccionario": tablero.diccionario,
             "tamano_mm": list(tablero.tamano_mm), "es_ejemplo": tablero.es_ejemplo,
             "convencion": "Origen centro; X derecha, Y fondo, Z arriba. Esquinas TL TR BR BL vistas desde +Z; borde superior hacia +Y.",
             "marcadores": {str(i): puntos.tolist() for i, puntos in tablero.marcadores.items()}}
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cargar_tablero(ruta):
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if datos.get("version") != 1 or datos.get("unidades") != "mm":
        raise ValueError("Se requiere tablero versión 1 con unidades mm")
    tamano = tuple(float(v) for v in datos["tamano_mm"])
    if len(tamano) != 2 or any(not np.isfinite(v) or v <= 0 for v in tamano):
        raise ValueError("Tamaño del tablero inválido")
    return Tablero(datos["diccionario"], datos["marcadores"], tamano,
                   es_ejemplo=bool(datos.get("es_ejemplo", True)))
