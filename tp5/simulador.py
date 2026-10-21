"""Mesa sintética con cámara y cartas independientes, y verdad geométrica.

Proyección pinhole, orientación ``mirar_a`` y renderizado de marcadores con
centros de píxel/supermuestreo adaptados de
``tp3-localizacion-homografica/localizacion/simulador.py``. La cámara móvil,
la distorsión y las transformaciones 3D se implementan para TP5.
"""
import cv2
import numpy as np

from .configuracion import CARTAS
from .deteccion import Detector
from .modelos import Calibracion, CuadroSintetico, Deteccion
from .pose import invertir_transformacion, transformar_puntos
from .tablero import crear_tablero_ejemplo, esquinas_cuadrado


def _mirar_a(centro, objetivo):
    z = np.asarray(objetivo, dtype=float) - centro
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.])
    if np.linalg.norm(x) < 1e-8:
        x = np.array([1., 0., 0.])
    else:
        x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.vstack([x, y, z])
    T[:3, 3] = -T[:3, :3] @ centro
    return T


class Simulador:
    def __init__(self, tablero=None, tamano=(1280, 720), semilla=0):
        self.tablero = tablero if tablero is not None else crear_tablero_ejemplo()
        self.tamano = tuple(tamano)
        w, h = tamano
        focal = 950. * w / 1280
        self.calibracion = Calibracion(np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.]]),
                                       np.zeros(5), self.tamano,
                                       metadatos={"origen": "sintetica", "unidades": "mm"})
        self._detector = Detector(self.tablero.diccionario)
        self._imagenes = {}
        self._mapa_clave = None
        self._mapas = None
        rng = np.random.default_rng(semilla)
        self._base = np.clip(np.full((h, w, 3), [57, 49, 42], dtype=float) +
                             rng.normal(0, .3, (h, w, 1)), 0, 255).astype(np.uint8)

    def _proyectar(self, puntos, T_cw, dist=None):
        rvec = cv2.Rodrigues(T_cw[:3, :3])[0]
        return cv2.projectPoints(np.asarray(puntos, dtype=float), rvec, T_cw[:3, 3],
                                 self.calibracion.K, self.calibracion.dist if dist is None else dist)[0].reshape(-1, 2)

    def _poligono(self, imagen, puntos, T_cw, color):
        proy = self._proyectar(puntos, T_cw, np.zeros(5))
        if np.isfinite(proy).all():
            cv2.fillPoly(imagen, [np.int32(np.round(proy * 16))], color,
                         cv2.LINE_AA, shift=4)

    def _pegar_marcador(self, imagen, identificador, puntos, T_cw):
        if identificador not in self._imagenes:
            self._imagenes[identificador] = self._detector.generar(identificador, 240)
        tinta = self._imagenes[identificador]
        n = tinta.shape[0]
        papel = (puntos - puntos.mean(axis=0)) * 1.36 + puntos.mean(axis=0)
        self._poligono(imagen, papel, T_cw, (250, 250, 250))
        esquinas = self._proyectar(puntos, T_cw, np.zeros(5))
        w, h = self.tamano
        x0, y0 = np.maximum(np.floor(esquinas.min(axis=0)).astype(int) - 2, [0, 0])
        x1, y1 = np.minimum(np.ceil(esquinas.max(axis=0)).astype(int) + 2, [w, h])
        if x1 <= x0 or y1 <= y0:
            return
        origen = np.float32([[0, 0], [n, 0], [n, n], [0, n]]) - .5
        H = cv2.getPerspectiveTransform(origen, np.float32(esquinas - [x0, y0]))
        tamano = (int(x1 - x0), int(y1 - y0))
        F = 8
        S = np.array([[F, 0, (F - 1) / 2], [0, F, (F - 1) / 2], [0, 0, 1.]])
        grande = cv2.warpPerspective(tinta, S @ H, (tamano[0] * F, tamano[1] * F),
                                     flags=cv2.INTER_NEAREST, borderValue=255)
        mascara = cv2.warpPerspective(np.full_like(tinta, 255), S @ H,
                                      (tamano[0] * F, tamano[1] * F), flags=cv2.INTER_NEAREST)
        cubre = cv2.resize(((grande < 128) & (mascara > 0)).astype(np.float32),
                           tamano, interpolation=cv2.INTER_AREA)[..., None]
        roi = imagen[y0:y1, x0:x1].astype(np.float32)
        imagen[y0:y1, x0:x1] = (roi * (1 - cubre) + 20 * cubre).astype(np.uint8)

    def _distorsionar(self, imagen):
        if np.allclose(self.calibracion.dist, 0):
            return imagen
        clave = (self.calibracion.K.tobytes(), self.calibracion.dist.tobytes())
        if clave != self._mapa_clave:
            w, h = self.tamano
            y, x = np.mgrid[0:h, 0:w]
            pixeles = np.stack([x, y], axis=-1).astype(np.float64).reshape(-1, 1, 2)
            # Para cada píxel distorsionado de salida buscamos el píxel ideal.
            ideal = cv2.undistortPoints(pixeles, self.calibracion.K, self.calibracion.dist,
                                        P=self.calibracion.K).reshape(h, w, 2)
            self._mapas = (ideal[..., 0].astype(np.float32), ideal[..., 1].astype(np.float32))
            self._mapa_clave = clave
        return cv2.remap(imagen, *self._mapas, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(57, 49, 42))

    def generar(self, frame_id, mover_camara=True, mover_cartas=True, ocultar_tablero=False):
        tiempo = frame_id / 30.
        tcam = tiempo if mover_camara else 0.
        centro = np.array([100 + 120 * np.sin(.55 * tcam),
                           -460 + 70 * np.sin(.38 * tcam),
                           900 + 80 * np.sin(.42 * tcam)])
        objetivo = np.array([30 * np.sin(.31 * tcam), 25 * np.sin(.43 * tcam), 0.])
        T_cw = _mirar_a(centro, objetivo)
        imagen = self._base.copy()
        wmm, hmm = self.tablero.tamano_mm
        superficie = np.array([[-wmm / 2 - 30, hmm / 2 + 30, 0],
                               [wmm / 2 + 30, hmm / 2 + 30, 0],
                               [wmm / 2 + 30, -hmm / 2 - 30, 0],
                               [-wmm / 2 - 30, -hmm / 2 - 30, 0]])
        self._poligono(imagen, superficie, T_cw, (92, 121, 112))
        geometria = {} if ocultar_tablero else dict(self.tablero.marcadores)
        cartas_T_wm = {}
        tcarta = tiempo if mover_cartas else 0.
        for i, config in CARTAS.items():
            T_wm = np.eye(4)
            signo = -1 if i == 10 else 1
            angulo = signo * (.2 + .35 * np.sin(.8 * tcarta))
            c, s = np.cos(angulo), np.sin(angulo)
            T_wm[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T_wm[:3, 3] = [signo * 100 + 35 * np.sin(.7 * tcarta),
                              signo * 35 + 38 * np.sin(.9 * tcarta), 3.]
            cartas_T_wm[i] = T_wm
            local_carta = np.array([[-36, 47, 0], [36, 47, 0], [36, -47, 0], [-36, -47, 0.]])
            self._poligono(imagen, transformar_puntos(T_wm, local_carta), T_cw, config.color_bgr)
            geometria[i] = transformar_puntos(T_wm, esquinas_cuadrado(config.lado_mm))
        detecciones = []
        for i, puntos in geometria.items():
            if np.any(transformar_puntos(T_cw, puntos)[:, 2] <= 0):
                continue
            esquinas = self._proyectar(puntos, T_cw)
            w, h = self.tamano
            if np.any(esquinas < 2) or np.any(esquinas >= np.array([w - 2, h - 2])):
                continue
            self._pegar_marcador(imagen, i, puntos, T_cw)
            detecciones.append(Deteccion(i, esquinas))
        imagen = self._distorsionar(imagen)
        return CuadroSintetico(imagen, frame_id, tiempo, T_cw, invertir_transformacion(T_cw),
                              cartas_T_wm, sorted(detecciones, key=lambda d: d.id))
