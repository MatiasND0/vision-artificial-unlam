"""Calibración pinhole OpenCV sobre imágenes crudas; geometría en milímetros.

Una calibración sintética verifica el software, no mide una cámara física. El
archivo conserva la resolución; no reescalar K implícitamente al cargarlo.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable
import warnings

import cv2
import numpy as np

from .modelos import Calibracion

MINIMO_VISTAS = 6
VERSION_CALIBRACION = 1


def validar_patron(patron: tuple[int, int], lado_mm: float) -> tuple[int, int]:
    """El patrón describe esquinas interiores (columnas, filas), no casilleros."""
    if len(patron) != 2 or any(isinstance(v, bool) or int(v) != v or v < 3 for v in patron):
        raise ValueError("El patrón requiere al menos 3×3 esquinas interiores enteras")
    if not math.isfinite(lado_mm) or lado_mm <= 0:
        raise ValueError("El lado de cada casillero debe ser positivo y estar medido en mm")
    return tuple(int(v) for v in patron)


def puntos_patron(patron: tuple[int, int], lado_mm: float) -> np.ndarray:
    """Esquinas interiores coplanares; el tamaño físico determina la escala."""
    columnas, filas = validar_patron(patron, lado_mm)
    puntos = np.zeros((columnas * filas, 3), np.float32)
    puntos[:, :2] = np.mgrid[0:columnas, 0:filas].T.reshape(-1, 2) * lado_mm
    return puntos


def detectar_patron(imagen: np.ndarray, patron: tuple[int, int]) -> np.ndarray | None:
    validar_patron(patron, 1.0)
    if imagen is None or imagen.size == 0:
        raise ValueError("La imagen del patrón está vacía")
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if imagen.ndim == 3 else imagen
    # SB incorpora refinamiento subpíxel y tolera perspectiva; fallback para
    # construcciones de OpenCV que solo expongan el detector clásico.
    if hasattr(cv2, "findChessboardCornersSB"):
        encontrado, esquinas = cv2.findChessboardCornersSB(
            gris, patron, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        encontrado, esquinas = cv2.findChessboardCorners(
            gris, patron, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if encontrado:
            esquinas = cv2.cornerSubPix(
                gris, esquinas, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    return esquinas.reshape(-1, 2) if encontrado else None


def _validar_tamano(tamano: tuple[int, int]) -> tuple[int, int]:
    if len(tamano) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in tamano):
        raise ValueError("La resolución debe ser (ancho, alto) con enteros positivos")
    return tuple(int(v) for v in tamano)


def _puntos_vista(puntos, dimension: int, indice: int) -> np.ndarray:
    puntos = np.asarray(puntos, dtype=np.float64)
    if puntos.size == 0 or puntos.shape[-1:] != (dimension,) or puntos.ndim not in (2, 3):
        raise ValueError(f"Vista {indice}: se esperan puntos de dimensión {dimension}")
    puntos = puntos.reshape(-1, dimension)
    if len(puntos) < 6 or not np.isfinite(puntos).all():
        raise ValueError(f"Vista {indice}: se requieren al menos 6 puntos finitos")
    if np.linalg.matrix_rank(puntos - puntos.mean(axis=0)) < 2:
        raise ValueError(f"Vista {indice}: los puntos no pueden ser colineales")
    return np.ascontiguousarray(puntos, dtype=np.float32)


def calibrar_desde_puntos(puntos_objeto, puntos_imagen, tamano,
                         metadatos=None) -> Calibracion:
    """Estima K y cinco coeficientes (k1,k2,p1,p2,k3), sin undistort previo.

    Exige seis vistas distintas para rechazar entradas evidentemente degeneradas.
    La diversidad restante se informa como recomendación, no como certificación
    de exactitud física. RMS bajo por sí solo no garantiza una buena calibración.
    """
    tamano = _validar_tamano(tamano)
    objetos = list(puntos_objeto)
    imagenes = list(puntos_imagen)
    if len(objetos) != len(imagenes) or len(objetos) < MINIMO_VISTAS:
        raise ValueError(f"Se requieren al menos {MINIMO_VISTAS} vistas con sus correspondencias")
    if metadatos is not None and not isinstance(metadatos, dict):
        raise ValueError("Los metadatos deben ser un objeto JSON")
    objetos = [_puntos_vista(p, 3, i) for i, p in enumerate(objetos)]
    imagenes = [_puntos_vista(p, 2, i) for i, p in enumerate(imagenes)]
    for i, (objeto, imagen) in enumerate(zip(objetos, imagenes)):
        if len(objeto) != len(imagen):
            raise ValueError(f"Vista {i}: cantidades distintas de puntos 3D y 2D")
        if np.any(imagen < 0) or np.any(imagen[:, 0] >= tamano[0]) or np.any(imagen[:, 1] >= tamano[1]):
            raise ValueError(f"Vista {i}: las esquinas deben pertenecer a la resolución declarada")
    firmas = {(p.shape, p.round(3).tobytes()) for p in imagenes}
    if len(firmas) < MINIMO_VISTAS:
        raise ValueError(f"Se requieren al menos {MINIMO_VISTAS} vistas distintas; evite repetir capturas")
    try:
        rms, K, dist, rotaciones, traslaciones = cv2.calibrateCamera(
            objetos, imagenes, tamano, None, None,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
    except cv2.error as error:
        raise ValueError("OpenCV no pudo calibrar: revisar geometría, esquinas y diversidad de vistas") from error
    errores = []
    inclinaciones = []
    for objeto, imagen, rotacion, traslacion in zip(objetos, imagenes, rotaciones, traslaciones):
        proyeccion, _ = cv2.projectPoints(objeto, rotacion, traslacion, K, dist)
        errores.append(float(np.sqrt(np.mean(np.sum((proyeccion.reshape(-1, 2) - imagen) ** 2, axis=1)))))
        normal = cv2.Rodrigues(rotacion)[0][:, 2]
        inclinaciones.append(float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))))
    centros = np.array([p.mean(axis=0) for p in imagenes]) / np.array(tamano)
    areas = np.array([cv2.contourArea(cv2.convexHull(p)) for p in imagenes]) / math.prod(tamano)
    recomendaciones = []
    if len(imagenes) < 15:
        recomendaciones.append("Capturar entre 15 y 25 vistas nítidas y diferentes")
    if np.ptp(centros[:, 0]) < 0.25 or np.ptp(centros[:, 1]) < 0.20:
        recomendaciones.append("Llevar el patrón al centro, los bordes y las esquinas de la imagen")
    if areas.max() / max(areas.min(), 1e-12) < 1.8:
        recomendaciones.append("Variar la distancia para que cambie el tamaño aparente del patrón")
    if max(inclinaciones) < 20.0 or np.ptp(inclinaciones) < 10.0:
        recomendaciones.append("Inclinar el patrón en ambos ejes, además de mostrarlo de frente")
    if rms > 1.0:
        recomendaciones.append("RMS superior a 1 px: revisar desenfoque, rigidez del patrón y vistas con error alto")
    info = dict(metadatos or {})
    info.update({
        "modelo": "opencv-pinhole-5", "vistas": len(imagenes),
        "rms_por_vista_px": errores, "recomendaciones": recomendaciones,
        "diversidad": {"rango_centros_normalizados": np.ptp(centros, axis=0).tolist(),
                       "fraccion_area_min_max": [float(areas.min()), float(areas.max())],
                       "inclinacion_min_max_grados": [min(inclinaciones), max(inclinaciones)]},
        "imagen": "cruda_sin_undistort", "unidades_objeto": "mm",
    })
    calibracion = Calibracion(K, dist, tamano, float(rms), info)
    if recomendaciones:
        warnings.warn("Calibración calculada. Recomendaciones: " + "; ".join(recomendaciones),
                      UserWarning, stacklevel=2)
    return calibracion


def guardar_calibracion(calibracion: Calibracion, ruta: str | Path) -> None:
    # Revalida incluso si alguien mutó los arrays después de construir el modelo.
    validada = Calibracion(calibracion.K, calibracion.dist, calibracion.tamano,
                           calibracion.rms_px, calibracion.metadatos)
    if not isinstance(validada.metadatos, dict):
        raise ValueError("Los metadatos deben ser un objeto JSON")
    datos = {"version": VERSION_CALIBRACION, "unidades": "mm",
             "K": validada.K.tolist(), "dist": validada.dist.tolist(),
             "tamano": [int(v) for v in validada.tamano], "rms_px": float(validada.rms_px),
             "metadatos": validada.metadatos}
    try:
        contenido = json.dumps(datos, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError("La calibración debe contener metadatos JSON finitos") from error
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(contenido, encoding="utf-8")


def cargar_calibracion(ruta: str | Path) -> Calibracion:
    try:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        if (not isinstance(datos, dict) or type(datos.get("version")) is not int
                or datos.get("version") != VERSION_CALIBRACION):
            raise ValueError("Versión de calibración no compatible; se espera versión 1")
        if datos.get("unidades") != "mm":
            raise ValueError("La calibración debe declarar unidades mm")
        if not isinstance(datos.get("metadatos"), dict):
            raise ValueError("Los metadatos deben ser un objeto JSON")
        # JSON permite NaN en Python por defecto; rechazarlo también en metadatos.
        json.dumps(datos, allow_nan=False)
        return Calibracion(datos["K"], datos["dist"], datos["tamano"],
                           datos["rms_px"], datos["metadatos"])
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError("Archivo de calibración incompleto o inválido") from error


def calibrar_desde_imagenes(rutas: Iterable[str | Path], patron=(9, 6),
                           lado_mm=25.0) -> Calibracion:
    objeto = puntos_patron(patron, lado_mm)
    puntos, objetos, aceptadas, descartadas = [], [], [], []
    tamano = None
    for ruta in rutas:
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if imagen is None:
            raise ValueError(f"No se pudo leer la imagen {ruta}")
        actual = (imagen.shape[1], imagen.shape[0])
        if tamano is not None and actual != tamano:
            raise ValueError(f"Resolución no uniforme: {ruta} mide {actual}; se esperaba {tamano}")
        tamano = actual
        esquinas = detectar_patron(imagen, patron)
        if esquinas is None:
            descartadas.append(str(ruta))
            continue
        puntos.append(esquinas)
        objetos.append(objeto.copy())
        aceptadas.append(str(ruta))
    if tamano is None:
        raise ValueError("No se recibieron imágenes para calibrar")
    return calibrar_desde_puntos(objetos, puntos, tamano, {
        "origen": "imagenes", "patron": "chessboard",
        "esquinas_interiores": list(patron), "lado_casillero_mm": lado_mm,
        "imagenes_aceptadas": aceptadas, "imagenes_sin_patron": descartadas,
    })


def guardar_patron_svg(ruta: str | Path, patron=(9, 6), lado_mm=25.0) -> None:
    columnas, filas = validar_patron(patron, lado_mm)
    margen = 10.0
    ancho, alto = (columnas + 1) * lado_mm + 2 * margen, (filas + 1) * lado_mm + 2 * margen + 10
    rectangulos = []
    for fila in range(filas + 1):
        for columna in range(columnas + 1):
            if (fila + columna) % 2 == 0:
                rectangulos.append(f'<rect x="{margen + columna * lado_mm:g}" y="{margen + fila * lado_mm:g}" '
                                   f'width="{lado_mm:g}" height="{lado_mm:g}" fill="black"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho:g}mm" height="{alto:g}mm" '
           f'viewBox="0 0 {ancho:g} {alto:g}">\n'
           f'<rect width="{ancho:g}" height="{alto:g}" fill="white"/>\n' + "\n".join(rectangulos) +
           f'\n<text x="{margen:g}" y="{alto - 5:g}" font-family="sans-serif" font-size="3">'
           f'TP5: {columnas}x{filas} esquinas interiores; casillero {lado_mm:g} mm. '
           'Imprimir al 100%; medir antes de calibrar.</text>\n</svg>\n')
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(svg, encoding="utf-8")
