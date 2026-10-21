"""Mallas AR con z-buffer EGL, sobre cámara cruda y sin corregirla dos veces.

Las mallas ya están en mundo/mm. El framebuffer contiene color premultiplicado
y alfa transparente; únicamente ese overlay vuelve al dominio distorsionado.
La iluminación es ambiental/difusa, sin el conjunto completo de materiales PBR.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .ar import _transformacion
from .modelos import Calibracion, PoseCamara

if TYPE_CHECKING:
    from .activos_3d import Modelo3D


_VERTEX = """
#version 330
uniform mat4 proyeccion;
uniform mat4 T_cw;
in vec3 posicion;
in vec3 normal;
in vec2 uv;
out vec3 normal_mundo;
out vec2 coordenada_uv;
void main() {
    gl_Position = proyeccion * T_cw * vec4(posicion, 1.0);
    normal_mundo = normal;
    coordenada_uv = uv;
}
"""

_FRAGMENT = """
#version 330
uniform sampler2D imagen_textura;
uniform vec4 color_material;
uniform int alpha_modo;
uniform float alpha_corte;
in vec3 normal_mundo;
in vec2 coordenada_uv;
out vec4 color;
void main() {
    vec4 base = texture(imagen_textura, coordenada_uv) * color_material;
    float alfa = base.a;
    if (alpha_modo == 1 && alfa < alpha_corte) discard;
    if (alpha_modo != 2) alfa = 1.0;
    vec3 luz = normalize(vec3(-0.3, -0.5, 1.0));
    float difusa = max(dot(normalize(normal_mundo), luz), 0.0);
    vec3 rgb = base.rgb * (0.35 + 0.65 * difusa);
    color = vec4(rgb * alfa, alfa);
}
"""


def matriz_proyeccion(calibracion: Calibracion, cercano_mm=1., lejano_mm=100000.):
    """OpenCV (+Z delante, Y abajo) → clip OpenGL, sin distorsión.

    El +0,5 al centro principal alinea centros enteros de píxel de OpenCV con
    centros de fragmento de OpenGL. Al leer la imagen se invierten sus filas.
    """
    cercano, lejano = float(cercano_mm), float(lejano_mm)
    if not np.isfinite([cercano, lejano]).all() or not 0 < cercano < lejano:
        raise ValueError("Los planos de recorte deben ser finitos y 0 < cercano < lejano")
    ancho, alto = calibracion.tamano
    fx, fy = calibracion.K[0, 0], calibracion.K[1, 1]
    cx, cy = calibracion.K[0, 2] + .5, calibracion.K[1, 2] + .5
    return np.array([[2*fx/ancho, 0, 2*cx/ancho - 1, 0],
                     [0, -2*fy/alto, 1 - 2*cy/alto, 0],
                     [0, 0, (lejano+cercano)/(lejano-cercano),
                      -2*lejano*cercano/(lejano-cercano)],
                     [0, 0, 1, 0]], dtype=np.float32)


def _mapas_overlay(calibracion):
    ancho, alto = calibracion.tamano
    y, x = np.indices((alto, ancho), dtype=np.float32)
    crudos = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
    ideales = cv2.undistortPoints(crudos, calibracion.K, calibracion.dist,
                                  P=calibracion.K).reshape(alto, ancho, 2)
    ideales[~np.isfinite(ideales)] = -1
    return np.ascontiguousarray(ideales[..., 0]), np.ascontiguousarray(ideales[..., 1])


class RenderizadorModelo:
    """Prepara todos los modelos una vez; cada cuadro usa solo la pose actual.

    Los materiales MASK descartan fragmentos antes de escribir profundidad.
    BLEND usa composición premultiplicada y orden por centro de cada parte;
    las transparencias que se intersectan no tienen orden exacto por triángulo.
    """
    def __init__(self, calibracion: Calibracion, modelos: Sequence[Modelo3D]):
        self.calibracion = calibracion
        self.modelos = tuple(modelos)
        if not self.modelos or any(not modelo.partes for modelo in self.modelos):
            raise ValueError("Se requiere al menos un modelo con mallas")
        self._indice = 0
        self._ctx = None
        self._recursos = []
        self._cerrado = False
        self._partes = []
        try:
            import moderngl
        except ImportError as exc:
            raise RuntimeError("El render 3D requiere moderngl y glcontext en el entorno del proyecto") from exc
        self._gl = moderngl
        try:
            self._ctx = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception as exc:
            self.cerrar()
            raise RuntimeError(f"No se pudo iniciar el render 3D con EGL/OpenGL 3.3: {exc}") from exc
        try:
            self._programa = self._guardar(self._ctx.program(vertex_shader=_VERTEX,
                                                          fragment_shader=_FRAGMENT))
            self._programa["imagen_textura"].value = 0
            self._programa["proyeccion"].write(matriz_proyeccion(calibracion).T.tobytes())
            self._color = self._guardar(self._ctx.texture(calibracion.tamano, 4, dtype="f1"))
            self._profundidad = self._guardar(self._ctx.depth_renderbuffer(calibracion.tamano))
            self._fbo = self._guardar(self._ctx.framebuffer([self._color], self._profundidad))
            self._mapa_x, self._mapa_y = _mapas_overlay(calibracion)
            for modelo in self.modelos:
                self._partes.append([self._cargar_parte(parte) for parte in modelo.partes])
        except Exception as exc:
            self.cerrar()
            raise RuntimeError(f"No se pudieron preparar las mallas para el render 3D: {exc}") from exc

    def _guardar(self, recurso):
        self._recursos.append(recurso)
        return recurso

    def _cargar_parte(self, parte):
        vertices = np.asarray(parte.vertices_mm, dtype=np.float32)
        normales = np.asarray(parte.normales, dtype=np.float32)
        uv = np.asarray(parte.uv, dtype=np.float32)
        indices = np.asarray(parte.triangulos, dtype=np.uint32)
        textura = np.asarray(parte.textura_rgba)
        color = np.asarray(parte.color_rgba, dtype=np.float32)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or normales.shape != vertices.shape or uv.shape != (len(vertices), 2)
                or indices.ndim != 2 or indices.shape[1] != 3 or not len(indices)
                or indices.max() >= len(vertices)
                or not all(np.isfinite(a).all() for a in (vertices, normales, uv, color))
                or color.shape != (4,) or np.any(color < 0) or np.any(color > 1)
                or textura.ndim != 3 or textura.shape[2] != 4 or textura.dtype != np.uint8
                or not textura.shape[0] or not textura.shape[1]
                or parte.alpha_modo not in ("OPAQUE", "MASK", "BLEND")
                or not np.isfinite(parte.alpha_corte) or not 0 <= parte.alpha_corte <= 1):
            raise ValueError("Malla, UV, textura RGBA o material inválidos")
        datos = np.ascontiguousarray(np.column_stack([vertices, normales, uv]), dtype=np.float32)
        vbo = self._guardar(self._ctx.buffer(datos.tobytes()))
        ibo = self._guardar(self._ctx.buffer(np.ascontiguousarray(indices).tobytes()))
        vao = self._guardar(self._ctx.vertex_array(self._programa,
                     [(vbo, "3f 3f 2f", "posicion", "normal", "uv")], ibo, index_element_size=4))
        tex = self._guardar(self._ctx.texture((textura.shape[1], textura.shape[0]), 4,
                                             np.ascontiguousarray(textura).tobytes(), alignment=1))
        tex.build_mipmaps()
        tex.filter = (self._gl.LINEAR_MIPMAP_LINEAR, self._gl.LINEAR)
        return {"vao": vao, "textura": tex, "color": tuple(float(c) for c in color),
                "modo": parte.alpha_modo, "corte": float(parte.alpha_corte),
                "centro": vertices.mean(axis=0)}

    @property
    def modelo_actual(self):
        return self.modelos[self._indice]

    def seleccionar(self, indice: int):
        if not isinstance(indice, Integral) or not 0 <= indice < len(self.modelos):
            raise IndexError("Índice de modelo 3D fuera de rango")
        self._indice = int(indice)

    def _dibujar_parte(self, parte):
        self._programa["color_material"].value = parte["color"]
        self._programa["alpha_modo"].value = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}[parte["modo"]]
        self._programa["alpha_corte"].value = parte["corte"]
        parte["textura"].use(0)
        parte["vao"].render(mode=self._gl.TRIANGLES)

    def dibujar(self, imagen: np.ndarray, pose: PoseCamara):
        if self._cerrado:
            raise RuntimeError("El renderizador 3D está cerrado")
        self.calibracion.validar_imagen(imagen)
        if imagen.ndim != 3 or imagen.shape[2] != 3 or imagen.dtype != np.uint8:
            raise ValueError("El render 3D requiere imagen BGR uint8")
        if not pose.valida:
            return imagen.copy()
        T_cw = _transformacion(pose.T_cw)
        try:
            self._fbo.use()
            self._fbo.depth_mask = True
            self._fbo.clear(0., 0., 0., 0., depth=1.)
            self._ctx.enable_only(self._gl.DEPTH_TEST | self._gl.BLEND)
            self._ctx.depth_func = "<"
            self._ctx.blend_func = self._gl.ONE, self._gl.ONE_MINUS_SRC_ALPHA
            self._programa["T_cw"].write(np.asarray(T_cw.T, dtype=np.float32).tobytes())
            partes = self._partes[self._indice]
            for parte in partes:
                if parte["modo"] != "BLEND":
                    self._dibujar_parte(parte)
            self._fbo.depth_mask = False
            transparentes = [p for p in partes if p["modo"] == "BLEND"]
            transparentes.sort(key=lambda p: (T_cw[:3, :3] @ p["centro"] + T_cw[:3, 3])[2],
                                reverse=True)
            for parte in transparentes:
                self._dibujar_parte(parte)
            self._fbo.depth_mask = True
            ancho, alto = self.calibracion.tamano
            rgba = np.frombuffer(self._fbo.read(components=4, alignment=1), dtype=np.uint8)
            rgba = rgba.reshape(alto, ancho, 4)[::-1].copy()
            overlay = cv2.remap(rgba, self._mapa_x, self._mapa_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            alfa = overlay[..., 3:4].astype(np.float32) / 255
            premultiplicado_bgr = overlay[..., :3][..., ::-1].astype(np.float32)
            return np.uint8(np.clip(np.rint(premultiplicado_bgr + imagen * (1 - alfa)), 0, 255))
        except Exception as exc:
            self.cerrar()
            raise RuntimeError(f"Falló el render 3D del modelo {self.modelo_actual.nombre}: {exc}") from exc

    def cerrar(self):
        if self._cerrado:
            return
        self._cerrado = True
        for recurso in reversed(self._recursos):
            try:
                recurso.release()
            except Exception:
                pass  # intentar liberar los restantes incluso tras un error de EGL
        self._recursos.clear()
        if self._ctx is not None:
            try:
                self._ctx.release()
            finally:
                self._ctx = None
