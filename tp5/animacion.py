"""Animación procedural y efectos métricos; no reproduce un rig ni clips glTF.

El dragón descargado no tiene skin. Su perfil usa únicamente las membranas altas
de las dos partes del torso: los brazos horizontales no son las alas. Las
coordenadas originales se conservan y cada cuadro se calcula desde esas copias.
"""
import unicodedata

import cv2
import numpy as np

from .ar import _segmento, proyectar_puntos


PERIODO_ALAS_S = 2.4
PERIODO_IDLE_S = 3.6


def perfil_modelo(modelo):
    texto = str(getattr(modelo, "nombre", "")) + " " + str(getattr(modelo, "ruta", ""))
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    if "dragon" in texto or "blue_eyes" in texto:
        return "dragon"
    if "magician" in texto or "mago" in texto:
        return "mago"
    return "generico"


class AnimadorModelo:
    """Flexión localizada de alas y respiración/flotación discreta en mm.

    El perfil de alas está limitado al asset Blue Eyes: partes que alcanzan la
    zona superior, por encima de cabeza y brazos. Se calcula una transición
    suave a cero en su raíz. Es deformación procedural, no animación esquelética.
    """
    def __init__(self, modelo):
        self.perfil = perfil_modelo(modelo)
        self.originales = [(np.array(p.vertices_mm, np.float32, copy=True),
                            np.array(p.normales, np.float32, copy=True)) for p in modelo.partes]
        todos = np.concatenate([v for v, _ in self.originales])
        minimo, maximo = todos.min(axis=0), todos.max(axis=0)
        self.base_z = float(minimo[2])
        self.altura = max(float(maximo[2]-minimo[2]), 1e-6)
        self.escala = float(modelo.tamano_mm) / 90.
        self.centro_xy = (minimo[:2] + maximo[:2]) / 2
        xy = todos[:, :2] - self.centro_xy
        _, ejes = np.linalg.eigh(xy.T @ xy)
        longitudinal = ejes[:, -1]
        self.lateral = np.array([longitudinal[1], -longitudinal[0]], dtype=np.float32)
        self.raiz_z = self.base_z + .62*self.altura
        self.raiz_lateral = .035*float(modelo.tamano_mm)
        self.mascaras_alas = []
        for v, _ in self.originales:
            if self.perfil == "dragon" and v[:, 2].max() > self.base_z + .9*self.altura:
                u = np.clip((v[:, 2] - (self.base_z + .67*self.altura)) / (.13*self.altura), 0, 1)
                peso = u*u*(3-2*u)
            else:
                peso = np.zeros(len(v), np.float32)
            self.mascaras_alas.append(peso.astype(np.float32))

    def deformar(self, tiempo_s):
        tiempo = float(tiempo_s)
        if not np.isfinite(tiempo):
            raise ValueError("El tiempo de animación debe ser finito")
        fase = 2*np.pi*tiempo/PERIODO_IDLE_S
        flotacion = .6*self.escala*(1-np.cos(fase))
        angulo = np.radians(22 + 18*np.sin(2*np.pi*tiempo/PERIODO_ALAS_S))
        resultado = []
        for (original, normal), peso in zip(self.originales, self.mascaras_alas):
            v, n = original.copy(), normal.copy()
            if self.perfil == "dragon" and np.any(peso):
                lateral = (original[:, :2] - self.centro_xy) @ self.lateral
                lado = np.where(lateral >= 0, 1., -1.)
                raiz = lado*self.raiz_lateral
                a = lado*angulo*peso
                c, s = np.cos(a), np.sin(a)
                x, z = lateral-raiz, original[:, 2]-self.raiz_z
                nuevo_lateral = raiz + c*x + s*z
                v[:, :2] += (nuevo_lateral-lateral)[:, None]*self.lateral
                v[:, 2] = self.raiz_z - s*x + c*z
                n_lateral = normal[:, :2] @ self.lateral
                nuevo_normal = c*n_lateral + s*normal[:, 2]
                n[:, :2] += (nuevo_normal-n_lateral)[:, None]*self.lateral
                n[:, 2] = -s*n_lateral + c*normal[:, 2]
                longitudes = np.linalg.norm(n, axis=1, keepdims=True)
                n /= np.maximum(longitudes, 1e-12)
            elif self.perfil == "mago":
                respiracion = 1 + .012*np.sin(fase)
                v[:, 2] = self.base_z + (original[:, 2]-self.base_z)*respiracion
                n[:, 2] /= respiracion
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            v[:, 2] += flotacion
            resultado.append((v, n))
        return resultado


def geometria_efectos(modelo, tiempo_s):
    """Anillos sobre Z=0,8 mm y partículas ascendentes en mundo/mm."""
    tiempo = float(tiempo_s)
    if not np.isfinite(tiempo):
        raise ValueError("El tiempo de efectos debe ser finito")
    radio = float(np.clip(.35*modelo.tamano_mm, 18, 45))
    angulos = np.linspace(0, 2*np.pi, 97)
    anillos = [np.column_stack([r*np.cos(angulos), r*np.sin(angulos),
                                np.full_like(angulos, .8)]) for r in (radio, .81*radio)]
    marcas = []
    for a in np.arange(8)*np.pi/4 + .45*tiempo:
        marcas.append(np.array([[.81*radio*np.cos(a), .81*radio*np.sin(a), .8],
                                 [radio*np.cos(a), radio*np.sin(a), .8]]))
    i = np.arange(12)
    ciclo = (i/12 + tiempo*.28) % 1
    a = i*2*np.pi/12 + tiempo*.35
    r = .78*radio + 1.5*np.sin(a*3 + tiempo)
    particulas = np.column_stack([r*np.cos(a), r*np.sin(a),
                                  1 + ciclo*.42*modelo.tamano_mm])
    return anillos, marcas, particulas, np.sin(np.pi*ciclo)


def dibujar_efectos(imagen, calibracion, pose, modelo, tiempo_s):
    """Copia BGR con glow; solo pose actual y K/dist sobre imagen cruda."""
    calibracion.validar_imagen(imagen)
    if imagen.ndim != 3 or imagen.shape[2] != 3 or imagen.dtype != np.uint8:
        raise ValueError("Los efectos requieren imagen BGR uint8")
    salida = imagen.copy()
    if not pose.valida:
        return salida
    anillos, marcas, particulas, brillos = geometria_efectos(modelo, tiempo_s)
    color = (255, 210, 65) if perfil_modelo(modelo) == "dragon" else (255, 85, 205)
    capa = np.zeros_like(imagen)
    for linea in anillos + marcas:
        pixeles, visibles = proyectar_puntos(linea, pose.T_cw, calibracion)
        if visibles.all() and np.max(np.abs(pixeles)) < 1e6:
            cv2.polylines(capa, [np.rint(np.clip(pixeles, -100000, 100000)).astype(np.int32)],
                           False, color, 1, cv2.LINE_AA)
        else:
            for j in range(len(linea)-1):
                if visibles[j] and visibles[j+1]:
                    _segmento(capa, pixeles[j], pixeles[j+1], color, 1)
    pixeles, visibles = proyectar_puntos(particulas, pose.T_cw, calibracion)
    extremos, ve = proyectar_puntos(particulas + [1., 0, 0], pose.T_cw, calibracion)
    alto, ancho = imagen.shape[:2]
    for i, (p, visible) in enumerate(zip(pixeles, visibles)):
        if not visible or not ve[i] or not 0 <= p[0] < ancho or not 0 <= p[1] < alto:
            continue
        radio_px = int(np.clip(np.linalg.norm(extremos[i]-p), 1, 5))
        tono = tuple(int(c*brillos[i]) for c in color)
        cv2.circle(capa, tuple(np.rint(p).astype(int)), radio_px, tono, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(capa, (0, 0), 2.2)
    return np.uint8(np.clip(imagen.astype(np.float32) + .55*capa + .75*glow, 0, 255))
