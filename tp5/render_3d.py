"""Mallas AR con z-buffer EGL, sobre cámara cruda y sin corregirla dos veces.

Las mallas están en mm, en mundo o en su ancla local según la T_cw recibida.
El framebuffer contiene color premultiplicado
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
from .animacion import AnimadorModelo, dibujar_efectos
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
    Todas las instancias comparten framebuffer y profundidad, con su T_cw propia.
    """
    def __init__(self, calibracion: Calibracion, modelos: Sequence[Modelo3D], *,
                 animar=False, efectos=False):
        self.calibracion = calibracion
        self.modelos = tuple(modelos)
        if not self.modelos or any(not modelo.partes for modelo in self.modelos):
            raise ValueError("Se requiere al menos un modelo con mallas")
        self._indice = 0
        self.animar = bool(animar)
        self.efectos = bool(efectos)
        self._inicio_tiempo = None
        self._animadores = ([AnimadorModelo(m) for m in self.modelos] if self.animar else [])
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
        vbo = self._guardar(self._ctx.buffer(datos.tobytes(), dynamic=self.animar))
        ibo = self._guardar(self._ctx.buffer(np.ascontiguousarray(indices).tobytes()))
        vao = self._guardar(self._ctx.vertex_array(self._programa,
                     [(vbo, "3f 3f 2f", "posicion", "normal", "uv")], ibo, index_element_size=4))
        tex = self._guardar(self._ctx.texture((textura.shape[1], textura.shape[0]), 4,
                                             np.ascontiguousarray(textura).tobytes(), alignment=1))
        tex.build_mipmaps()
        tex.filter = (self._gl.LINEAR_MIPMAP_LINEAR, self._gl.LINEAR)
        return {"vao": vao, "vbo": vbo, "uv": uv, "textura": tex,
                "color": tuple(float(c) for c in color),
                "modo": parte.alpha_modo, "corte": float(parte.alpha_corte),
                "centro": vertices.mean(axis=0)}

    @property
    def modelo_actual(self):
        return self.modelos[self._indice]

    def seleccionar(self, indice: int):
        self._indice = self._validar_indice(indice)

    def _validar_indice(self, indice):
        if not isinstance(indice, Integral) or not 0 <= indice < len(self.modelos):
            raise IndexError("Índice de modelo 3D fuera de rango")
        return int(indice)

    def _dibujar_parte(self, parte):
        self._programa["color_material"].value = parte["color"]
        self._programa["alpha_modo"].value = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}[parte["modo"]]
        self._programa["alpha_corte"].value = parte["corte"]
        parte["textura"].use(0)
        parte["vao"].render(mode=self._gl.TRIANGLES)

    def dibujar(self, imagen: np.ndarray, pose: PoseCamara):
        """Modo anterior: una instancia del modelo seleccionado con 1/2."""
        return self.dibujar_instancias(imagen, [(self._indice, pose)])

    def dibujar_instancias(self, imagen: np.ndarray,
                          instancias: Sequence[tuple[int, PoseCamara]]):
        """Renderiza una escena con z-buffer compartido y devuelve copia BGR.

        Cada T_cw transforma las coordenadas del modelo/ancla a la MISMA cámara;
        no declara que una pose de ancla sea la localización mundial de cámara.
        Las poses inválidas se omiten; nunca se reutiliza ultima_T_wc.
        """
        if self._cerrado:
            raise RuntimeError("El renderizador 3D está cerrado")
        self.calibracion.validar_imagen(imagen)
        if imagen.ndim != 3 or imagen.shape[2] != 3 or imagen.dtype != np.uint8:
            raise ValueError("El render 3D requiere imagen BGR uint8")
        # Completar la validación antes de modificar framebuffer, buffers o reloj.
        observadas = []
        instantes = []
        for indice, pose in instancias:
            indice = self._validar_indice(indice)
            if not isinstance(pose, PoseCamara):
                raise TypeError("Cada instancia requiere una PoseCamara")
            instante = 0.
            if self.animar or self.efectos:
                instante = float(pose.tiempo_s)
                if not np.isfinite(instante):
                    raise ValueError("El cuadro debe tener un tiempo finito para animación y efectos")
                instantes.append(instante)
            if pose.valida:
                T_cw = _transformacion(pose.T_cw).copy()
                observadas.append((indice, pose, T_cw, instante))
        if self._inicio_tiempo is None and instantes:
            self._inicio_tiempo = min(instantes)
        if not observadas:
            return imagen.copy()
        try:
            trabajos = []
            animaciones = {}
            fondo = imagen
            for indice, pose, T_cw, instante in observadas:
                tiempo = max(0., instante-self._inicio_tiempo) if instantes else 0.
                if self.efectos:
                    fondo = dibujar_efectos(fondo, self.calibracion, pose,
                                           self.modelos[indice], tiempo)
                partes = self._partes[indice]
                clave = (indice, tiempo)
                if self.animar and clave not in animaciones:
                    animaciones[clave] = []
                    deformadas = self._animadores[indice].deformar(tiempo)
                    for parte, (vertices, normales) in zip(partes, deformadas):
                        datos = np.ascontiguousarray(
                            np.column_stack([vertices, normales, parte["uv"]]), dtype=np.float32)
                        animaciones[clave].append((datos.tobytes(), vertices.mean(axis=0)))
                for numero, parte in enumerate(partes):
                    datos, centro = (animaciones[clave][numero] if self.animar
                                     else (None, parte["centro"]))
                    profundidad = float((T_cw[:3, :3] @ centro + T_cw[:3, 3])[2])
                    trabajos.append({"parte": parte, "T_cw": T_cw, "datos": datos,
                                     "profundidad": profundidad, "indice": indice,
                                     "numero": numero})
            self._fbo.use()
            self._fbo.depth_mask = True
            self._fbo.clear(0., 0., 0., 0., depth=1.)
            self._ctx.enable_only(self._gl.DEPTH_TEST | self._gl.BLEND)
            self._ctx.depth_func = "<"
            self._ctx.blend_func = self._gl.ONE, self._gl.ONE_MINUS_SRC_ALPHA
            for trabajo in trabajos:
                if trabajo["parte"]["modo"] != "BLEND":
                    self._dibujar_trabajo(trabajo)
            self._fbo.depth_mask = False
            transparentes = [t for t in trabajos if t["parte"]["modo"] == "BLEND"]
            transparentes.sort(key=lambda t: (t["profundidad"], t["indice"], t["numero"]),
                                reverse=True)
            for trabajo in transparentes:
                self._dibujar_trabajo(trabajo)
            self._fbo.depth_mask = True
            ancho, alto = self.calibracion.tamano
            rgba = np.frombuffer(self._fbo.read(components=4, alignment=1), dtype=np.uint8)
            rgba = rgba.reshape(alto, ancho, 4)[::-1].copy()
            overlay = cv2.remap(rgba, self._mapa_x, self._mapa_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            alfa = overlay[..., 3:4].astype(np.float32) / 255
            premultiplicado_bgr = overlay[..., :3][..., ::-1].astype(np.float32)
            return np.uint8(np.clip(np.rint(premultiplicado_bgr + fondo * (1 - alfa)), 0, 255))
        except Exception as exc:
            self.cerrar()
            nombres = ", ".join(self.modelos[i].nombre for i in dict.fromkeys(o[0] for o in observadas))
            raise RuntimeError(f"Falló el render 3D de las instancias {nombres}: {exc}") from exc

    def _dibujar_trabajo(self, trabajo):
        parte = trabajo["parte"]
        if trabajo["datos"] is not None:
            # Un mismo VAO puede aparecer con otra pose/fase en este cuadro.
            # Escribir justo antes de dibujar evita compartir su deformación.
            parte["vbo"].write(trabajo["datos"])
        self._programa["T_cw"].write(np.asarray(trabajo["T_cw"].T, dtype=np.float32).tobytes())
        self._dibujar_parte(parte)

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
