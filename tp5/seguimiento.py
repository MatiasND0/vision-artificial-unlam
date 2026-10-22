"""Recuperación visual de ArUco sin reutilizar una pose de un cuadro anterior.

Primero decodifica normalmente, después busca en una región ampliada y, durante
un máximo de 0,3 s, puede medir el desplazamiento con flujo óptico. El flujo
exige consistencia de ida/vuelta, homografía robusta y el patrón binario actual.
Las esquinas nunca se promedian entre imágenes de una cámara móvil.
"""
import time

import cv2
import numpy as np

from .deteccion import Detector, refinar_por_bordes
from .modelos import Deteccion


class DetectorContinuo(Detector):
    def __init__(self, diccionario="4x4_50", *, ids_fijos=None, ids_moviles=(), max_flujo_s=.3):
        super().__init__(diccionario)
        if not np.isfinite(max_flujo_s) or not 0 < max_flujo_s <= .5:
            raise ValueError("El intervalo de flujo debe estar entre 0 y 0,5 segundos")
        self.ids_fijos = None if ids_fijos is None else set(ids_fijos)
        self.ids_seguimiento = None if self.ids_fijos is None else self.ids_fijos | set(ids_moviles)
        self.max_flujo_s = float(max_flujo_s)
        self._historial = {}
        self._gris_anterior = None
        self._tiempo_anterior = None
        self._patrones = {}
        self.fuentes = {}
        self.diagnostico = {}

    def _patron_actual(self, gris, puntos, identificador):
        n = self.diccionario.markerSize + 2
        lado = n * 12
        destino = np.float32([[0, 0], [lado-1, 0], [lado-1, lado-1], [0, lado-1]])
        H = cv2.getPerspectiveTransform(np.float32(puntos), destino)
        plano = cv2.warpPerspective(gris, H, (lado, lado))
        valores = np.array([[np.median(plano[y*12+4:y*12+8, x*12+4:x*12+8])
                             for x in range(n)] for y in range(n)])
        if identificador not in self._patrones:
            original = self.generar(identificador, lado)
            self._patrones[identificador] = original[6::12, 6::12] > 128
        esperado = self._patrones[identificador]
        negro = float(np.median(valores[~esperado]))
        blanco = float(np.median(valores[esperado]))
        if blanco - negro < 35:
            return False
        leido = valores > (blanco + negro) / 2
        # Identidad exacta en los datos; tolerar como máximo un módulo del borde.
        return (np.array_equal(leido[1:-1, 1:-1], esperado[1:-1, 1:-1])
                and np.count_nonzero(leido != esperado) <= 1)

    def _roi(self, gris, puntos, identificador):
        lado = np.linalg.norm(puntos - np.roll(puntos, -1, axis=0), axis=1).mean()
        margen = max(35., lado * .75)
        x0, y0 = np.maximum(np.floor(puntos.min(axis=0) - margen).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(puntos.max(axis=0) + margen).astype(int),
                            [gris.shape[1], gris.shape[0]])
        roi = gris[y0:y1, x0:x1]
        if not roi.size:
            return None
        grande = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        esquinas, ids, _ = self._detector.detectMarkers(grande)
        if ids is not None:
            for esq, id_actual in zip(esquinas, ids.flatten()):
                if int(id_actual) == identificador:
                    p = (esq.reshape(4, 2) + .5) / 3 - .5 + [x0, y0]
                    return refinar_por_bordes(gris.astype(np.float32), np.float32(p),
                                               self.diccionario.markerSize + 2)
        return None

    def _flujo(self, gris, puntos, identificador):
        mascara = np.zeros_like(self._gris_anterior)
        cv2.fillConvexPoly(mascara, np.rint(puntos).astype(np.int32), 255)
        anteriores = cv2.goodFeaturesToTrack(self._gris_anterior, 45, .015, 3,
                                              mask=mascara, blockSize=3)
        if anteriores is None or len(anteriores) < 6:
            return None
        opciones = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        actuales, ida, _ = cv2.calcOpticalFlowPyrLK(self._gris_anterior, gris,
                                                   anteriores, None, **opciones)
        if actuales is None:
            return None
        retorno, vuelta, _ = cv2.calcOpticalFlowPyrLK(gris, self._gris_anterior,
                                                      actuales, None, **opciones)
        if retorno is None:
            return None
        validas = ((ida.ravel() != 0) & (vuelta.ravel() != 0)
                   & (np.linalg.norm(retorno - anteriores, axis=2).ravel() < .6))
        a, b = anteriores[validas, 0], actuales[validas, 0]
        if len(a) < 6:
            return None
        H, mascara_h = cv2.findHomography(a, b, cv2.RANSAC, 1., maxIters=200,
                                          confidence=.99)
        if H is None or mascara_h is None:
            return None
        dentro = mascara_h.ravel().astype(bool)
        if dentro.sum() < 6 or dentro.mean() < .7:
            return None
        proy = cv2.perspectiveTransform(a[dentro, None], H)[:, 0]
        if np.sqrt(np.mean(np.sum((proy - b[dentro])**2, axis=1))) > .7:
            return None
        nuevos = cv2.perspectiveTransform(np.float32(puntos)[None], H)[0]
        if (not np.isfinite(nuevos).all() or not cv2.isContourConvex(nuevos)
                or np.any(nuevos < 2)
                or np.any(nuevos >= np.array([gris.shape[1]-2, gris.shape[0]-2]))):
            return None
        area_anterior = abs(cv2.contourArea(np.float32(puntos)))
        area_actual = abs(cv2.contourArea(nuevos))
        if area_anterior < 36 or not .65 < area_actual / area_anterior < 1.55:
            return None
        if not self._patron_actual(gris, nuevos, identificador):
            return None
        return nuevos

    def detectar(self, imagen, tiempo_s=None):
        instante = time.monotonic() if tiempo_s is None else float(tiempo_s)
        if not np.isfinite(instante):
            raise ValueError("El tiempo de seguimiento debe ser finito")
        gris = imagen if imagen.ndim == 2 else cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        if (self._tiempo_anterior is not None and
                (instante <= self._tiempo_anterior or instante - self._tiempo_anterior > .5
                 or gris.shape != self._gris_anterior.shape)):
            self._historial.clear()
            self._gris_anterior = None
        detecciones = super().detectar(imagen)
        self.fuentes = {d.id: "aruco" for d in detecciones}
        actuales = {d.id: d for d in detecciones}
        decodificadas = set(actuales)
        for id_fijo, (puntos, ultimo_id) in list(self._historial.items()):
            if id_fijo in actuales:
                continue
            if instante - ultimo_id > .5:
                continue
            nuevos = self._roi(gris, puntos, id_fijo)
            fuente = "aruco_roi"
            if nuevos is not None:
                decodificadas.add(id_fijo)
            elif (self._gris_anterior is not None
                    and instante - ultimo_id <= self.max_flujo_s):
                nuevos = self._flujo(gris, puntos, id_fijo)
                fuente = "flujo_verificado"
            if nuevos is not None:
                actuales[id_fijo] = Deteccion(id_fijo, nuevos)
                self.fuentes[id_fijo] = fuente
        historial = {}
        for id_actual, deteccion in actuales.items():
            if self.ids_seguimiento is None or id_actual in self.ids_seguimiento:
                ultimo_id = (instante if id_actual in decodificadas
                             else self._historial[id_actual][1])
                historial[id_actual] = (deteccion.esquinas.copy(), ultimo_id)
        self._historial = historial
        self._gris_anterior = gris.copy()
        self._tiempo_anterior = instante
        self.diagnostico = {"fuentes": {str(k): v for k, v in self.fuentes.items()},
                            "max_flujo_s": self.max_flujo_s}
        return [actuales[k] for k in sorted(actuales)]
