"""Detección ArUco por cuadro, sin acumular esquinas de una cámara móvil.

El refinamiento por intersección de bordes y la generación se adaptaron de
``tp3-localizacion-homografica/localizacion/deteccion.py`` (proyecto TP3).
"""
import re

import cv2
import numpy as np

from .modelos import Deteccion


def obtener_diccionario(nombre="4x4_50"):
    if not re.fullmatch(r"(?:[4-7]x[4-7]_(?:50|100|250|1000)|aruco_original)", nombre.lower()):
        raise ValueError(f"Diccionario ArUco desconocido: {nombre}")
    atributo = "DICT_" + nombre.upper()
    valor = getattr(cv2.aruco, atributo, None)
    if valor is None:
        raise ValueError(f"Diccionario ArUco no disponible: {nombre}")
    return cv2.aruco.getPredefinedDictionary(valor)


class Detector:
    """Detector original con rescate de umbral experimental y optativo.

    ``rescate_umbral=True`` permite comparar otra escala de umbral en cuadros
    sin marcadores. Puede recuperar IDs con esquinas imprecisas y empeorar la
    continuidad de PnP; por eso permanece deshabilitado por defecto.
    """
    def __init__(self, diccionario="4x4_50", refinar_bordes=True, rescate_umbral=False):
        self.nombre_diccionario = diccionario
        self.diccionario = obtener_diccionario(diccionario)
        self.refinar_bordes = refinar_bordes
        parametros = cv2.aruco.DetectorParameters()
        parametros.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self.diccionario, parametros)
        self._detector_rescate = None
        if rescate_umbral:
            parametros_rescate = cv2.aruco.DetectorParameters()
            parametros_rescate.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            parametros_rescate.adaptiveThreshWinSizeMin = 5
            parametros_rescate.adaptiveThreshWinSizeMax = 35
            parametros_rescate.adaptiveThreshWinSizeStep = 10
            self._detector_rescate = cv2.aruco.ArucoDetector(self.diccionario, parametros_rescate)

    def detectar(self, imagen):
        gris = imagen if imagen.ndim == 2 else cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = self._detector.detectMarkers(gris)
        # Conservar la pasada original; otra escala de umbral solo rescata
        # cuadros sin marcadores, sin relajar su identificación ni sus filtros.
        if ids is None and self._detector_rescate is not None:
            esquinas, ids, _ = self._detector_rescate.detectMarkers(gris)
        if ids is None:
            return []
        g = gris.astype(np.float32) if self.refinar_bordes else None
        salida = []
        for esquinas_i, identificador in zip(esquinas, ids.flatten()):
            puntos = esquinas_i.reshape(4, 2)
            if self.refinar_bordes:
                puntos = refinar_por_bordes(g, puntos, self.diccionario.markerSize + 2)
            salida.append(Deteccion(int(identificador), puntos))
        return sorted(salida, key=lambda d: d.id)

    def generar(self, id_marcador, lado_px, borde_px=0):
        if not 0 <= int(id_marcador) < len(self.diccionario.bytesList):
            raise ValueError("ID fuera del diccionario")
        imagen = cv2.aruco.generateImageMarker(self.diccionario, int(id_marcador), int(lado_px))
        if borde_px:
            imagen = cv2.copyMakeBorder(imagen, borde_px, borde_px, borde_px, borde_px,
                                       cv2.BORDER_CONSTANT, value=255)
        return imagen


def _recta(p, q):
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def refinar_por_bordes(gris, esquinas, modulos=6, muestras=30):
    """Refina solo los bordes observados en ESTE cuadro; adaptación de TP3."""
    g = gris if gris.dtype == np.float32 else gris.astype(np.float32)
    centro = esquinas.mean(axis=0)
    lado = np.linalg.norm(esquinas - np.roll(esquinas, -1, axis=0), axis=1).mean()
    alcance = float(np.clip(0.4 * lado / modulos, 1.5, 4.0))
    desplazamientos = np.linspace(-alcance, alcance, int(alcance * 8) + 1)
    t = np.linspace(0.12, 0.88, muestras)
    rectas = []
    for i in range(4):
        a, b = esquinas[i], esquinas[(i + 1) % 4]
        d = b - a
        longitud = np.linalg.norm(d)
        if longitud < 1e-6:
            return esquinas
        u = d / longitud
        normal = np.array([-u[1], u[0]])
        if np.dot(normal, (a + b) / 2 - centro) < 0:
            normal = -normal
        puntos = a + t[:, None] * d
        grilla = puntos[:, None, :] + desplazamientos[None, :, None] * normal
        perfiles = cv2.remap(g, grilla[..., 0].astype(np.float32),
                             grilla[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        borde = []
        for p, perfil in zip(puntos, perfiles):
            negro, blanco = perfil[:3].mean(), perfil[-3:].mean()
            if blanco - negro < 25:
                continue
            nivel = (negro + blanco) / 2
            k = int(np.argmax(perfil >= nivel))
            if k == 0:
                continue
            salto = perfil[k] - perfil[k - 1]
            if abs(salto) < 1e-8:
                continue
            f = (nivel - perfil[k - 1]) / salto
            s = desplazamientos[k - 1] + f * (desplazamientos[k] - desplazamientos[k - 1])
            borde.append(p + s * normal)
        if len(borde) < muestras // 3:
            return esquinas
        vx, vy, x0, y0 = cv2.fitLine(np.float32(borde), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        rectas.append(_recta((x0, y0), (x0 + vx, y0 + vy)))
    refinadas = []
    for i in range(4):
        x = np.cross(rectas[(i - 1) % 4], rectas[i])
        if abs(x[2]) < 1e-10:
            return esquinas
        refinadas.append(x[:2] / x[2])
    refinadas = np.float32(refinadas)
    if not np.isfinite(refinadas).all() or np.abs(refinadas - esquinas).max() > 3.0:
        return esquinas
    return refinadas
