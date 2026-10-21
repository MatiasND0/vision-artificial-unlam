"""Vista externa del mundo, en mm, desacoplada de la cámara estimada.

Open3D se importa únicamente al habilitar la ventana. La misma geometría puede
inspeccionarse sin pantalla con ``datos_escena`` y ``renderizar_vista_externa``.
"""
from collections import deque
from dataclasses import dataclass
import importlib

import cv2
import numpy as np

from .ar import _transformacion, geometria_carta
from .modelos import Calibracion, EstadoMundo, Tablero


@dataclass
class _Lineas:
    puntos: np.ndarray
    indices: np.ndarray
    colores: np.ndarray  # RGB normalizado, convención de Open3D


def _lineas(puntos=(), indices=(), color=(.5, .5, .5)):
    puntos = np.asarray(puntos, dtype=float).reshape(-1, 3)
    indices = np.asarray(indices, dtype=np.int32).reshape(-1, 2)
    colores = np.broadcast_to(color, (len(indices), 3)).copy()
    return _Lineas(puntos, indices, colores)


def _unir(*geometrias):
    puntos, indices, colores, offset = [], [], [], 0
    for g in geometrias:
        puntos.extend(g.puntos)
        indices.extend(g.indices + offset)
        colores.extend(g.colores)
        offset += len(g.puntos)
    g = _lineas(puntos, indices)
    g.colores = np.asarray(colores).reshape(-1, 3)
    return g


def _transformar(T, puntos):
    T = _transformacion(T)
    return np.asarray(puntos) @ T[:3, :3].T + T[:3, 3]


def puntos_frustum(calibracion: Calibracion, T_wc, profundidad_mm=120.):
    """Origen y cuatro rayos de imagen, llevados al mundo por T_wc.

    La distancia al plano del frustum se mide sobre el +Z óptico de OpenCV.
    Se consideran los intrínsecos y la distorsión al recuperar los rayos.
    """
    if not np.isfinite(profundidad_mm) or profundidad_mm <= 0:
        raise ValueError("La profundidad del frustum debe ser positiva, en mm")
    w, h = calibracion.tamano
    esquinas = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=float)
    rayos = cv2.undistortPoints(esquinas.reshape(-1, 1, 2), calibracion.K,
                                calibracion.dist).reshape(-1, 2)
    plano = np.column_stack([rayos, np.ones(4)]) * profundidad_mm
    return _transformar(T_wc, np.vstack([np.zeros(3), plano]))


def _ejes(T, largo_mm):
    puntos = _transformar(T, [[0, 0, 0], [largo_mm, 0, 0],
                              [0, largo_mm, 0], [0, 0, largo_mm]])
    g = _lineas(puntos, [[0, 1], [0, 2], [0, 3]])
    g.colores = np.array([[1, .15, .15], [.15, 1, .15], [.2, .4, 1]])
    return g


def _puntear(g):
    segmentos = []
    for (a, b), color in zip(g.indices, g.colores):
        for t in np.arange(0, 1, .2):
            p = g.puntos[a] * (1-t) + g.puntos[b] * t
            q = g.puntos[a] * (1-t-.1) + g.puntos[b] * (t+.1)
            segmentos.append(_lineas([p, q], [[0, 1]], color))
    return _unir(*segmentos)


class Visualizador3D:
    """Tablero fijo, cámara, dirección +Z y trayectoria de hasta 600 poses.

    El observador externo se fija al crear la ventana y nunca sigue a T_wc.
    Una pérdida conserva, si existe, una cámara naranja punteada; no agrega
    puntos a la trayectoria y corta su conexión con la siguiente medición.
    """
    def __init__(self, tablero: Tablero, calibracion: Calibracion, habilitado=True):
        self.tablero, self.calibracion = tablero, calibracion
        self.habilitado = bool(habilitado)
        self._trayectoria = deque(maxlen=600)
        self._segmento = 0
        self._seguia = False
        self._ultimo_frame = None
        self._ultima_T_wc = None
        self._estado = None
        self._cerrado = False
        self._vis = None
        self._estatica = self._crear_tablero()
        self._dinamica = _lineas()
        self.datos_escena = {"camara_valida": False, "T_wc_actual": None,
                             "T_wc_mostrada": None, "frustum": np.empty((0, 3)),
                             "direccion": np.empty((0, 3)), "cartas": {},
                             "trayectoria": np.empty((0, 3)),
                             "tramos_trayectoria": np.empty((0, 2), dtype=int)}
        if self.habilitado:
            self._abrir_ventana()

    def _crear_tablero(self):
        w, h = self.tablero.tamano_mm
        if not np.isfinite([w, h]).all() or min(w, h) <= 0:
            raise ValueError("El tamaño del tablero debe ser positivo, en mm")
        borde = [[-w/2, -h/2, 0], [w/2, -h/2, 0],
                 [w/2, h/2, 0], [-w/2, h/2, 0]]
        gs = [_lineas(borde, [[0, 1], [1, 2], [2, 3], [3, 0]], (.35, .7, .4)),
              _ejes(np.eye(4), 130)]
        for x in np.arange(np.ceil(-w/200)*100, w/2 + 1, 100):
            gs.append(_lineas([[x, -h/2, 0], [x, h/2, 0]], [[0, 1]], (.2, .3, .3)))
        for y in np.arange(np.ceil(-h/200)*100, h/2 + 1, 100):
            gs.append(_lineas([[-w/2, y, 0], [w/2, y, 0]], [[0, 1]], (.2, .3, .3)))
        for esquinas in self.tablero.marcadores.values():
            gs.append(_lineas(esquinas, [[0, 1], [1, 2], [2, 3], [3, 0]], (.85, .85, .85)))
        return _unir(*gs)

    def _abrir_ventana(self):
        try:
            self._o3d = importlib.import_module("open3d")
        except ImportError as exc:
            raise RuntimeError("Open3D no está disponible. Usar modo sin 3D para las "
                               "pruebas sintéticas; instalar Open3D en un Python compatible "
                               "para la ventana interactiva.") from exc
        self._vis = self._o3d.visualization.Visualizer()
        if not self._vis.create_window("TP5 | mundo [mm] | X rojo Y verde Z azul | perdida: naranja punteada",
                                       width=1100, height=800):
            self._vis = None
            raise RuntimeError("Open3D no pudo crear una ventana; ejecutar con pantalla o en modo sin 3D")
        self._g_estatica = self._convertir_o3d(self._estatica)
        # Open3D puede omitir el renderer de una LineSet vacía al agregarla.
        # Semilla degenerada invisible; actualizar() sustituye toda su geometría.
        semilla = _lineas(np.zeros((2, 3)), [[0, 1]], (.035, .045, .055))
        self._g_dinamica = self._convertir_o3d(semilla)
        self._vis.add_geometry(self._g_estatica)
        self._vis.add_geometry(self._g_dinamica, reset_bounding_box=False)
        opciones = self._vis.get_render_option()
        opciones.background_color = np.array([.035, .045, .055])
        opciones.line_width = 2.
        vista = self._vis.get_view_control()
        vista.set_lookat([0, 0, 220])
        vista.set_front([.35, -.75, .65])
        vista.set_up([0, 0, 1])
        vista.set_zoom(.40)

    def _convertir_o3d(self, geometria, destino=None):
        destino = self._o3d.geometry.LineSet() if destino is None else destino
        destino.points = self._o3d.utility.Vector3dVector(geometria.puntos)
        destino.lines = self._o3d.utility.Vector2iVector(geometria.indices)
        destino.colors = self._o3d.utility.Vector3dVector(geometria.colores)
        return destino

    def actualizar(self, estado: EstadoMundo) -> bool:
        if self._cerrado:
            return False
        self._estado = estado
        pose = estado.camara
        nueva = pose.frame_id != self._ultimo_frame
        if nueva and pose.valida:
            if not self._seguia or (self._ultimo_frame is not None and pose.frame_id != self._ultimo_frame + 1):
                self._segmento += 1
            self._ultima_T_wc = _transformacion(pose.T_wc).copy()
            self._trayectoria.append((self._ultima_T_wc[:3, 3].copy(), self._segmento))
        self._seguia = pose.valida
        self._ultimo_frame = pose.frame_id
        mostrada = pose.T_wc if pose.valida else pose.ultima_T_wc
        if mostrada is None and not pose.valida:
            mostrada = self._ultima_T_wc
        puntos_tray = np.array([p for p, _ in self._trayectoria]).reshape(-1, 3)
        tramos = [[i-1, i] for i in range(1, len(self._trayectoria))
                  if self._trayectoria[i-1][1] == self._trayectoria[i][1]]
        trayectoria = _lineas(puntos_tray, tramos, (.25, .85, .95))
        gs, cartas = [trayectoria], {}
        frustum, direccion = np.empty((0, 3)), np.empty((0, 3))
        if mostrada is not None:
            mostrada = _transformacion(mostrada)
            frustum = puntos_frustum(self.calibracion, mostrada)
            direccion = _transformar(mostrada, [[0, 0, 0], [0, 0, 180]])
            color = (.1, .9, 1.) if pose.valida else (1., .35, .08)
            camara = _unir(_lineas(frustum, [[0, 1], [0, 2], [0, 3], [0, 4],
                                             [1, 2], [2, 3], [3, 4], [4, 1]], color),
                           _lineas(direccion, [[0, 1]], color), _ejes(mostrada, 65))
            if not pose.valida:
                camara.colores[:] = color
                camara = _puntear(camara)
            gs.append(camara)
        # Solo componer cartas actuales cuando hay pose mundial actual.
        if pose.valida:
            for carta in estado.cartas:
                if carta.frame_id != pose.frame_id:
                    continue
                T_wm = _transformacion(pose.T_wc) @ _transformacion(carta.T_cm)
                cartas[carta.id] = T_wm.copy()
                g = geometria_carta(40.)  # representación simbólica; escala rotulada en mm
                gs.extend([_ejes(T_wm, 45),
                           _lineas(_transformar(T_wm, g["vertical"]),
                                   [[0, 1], [1, 2], [2, 3], [3, 0]], (.8, .5, 1.))])
        self._dinamica = _unir(*gs)
        self.datos_escena = {"camara_valida": pose.valida,
                             "T_wc_actual": None if not pose.valida else np.asarray(pose.T_wc).copy(),
                             "T_wc_mostrada": None if mostrada is None else mostrada.copy(),
                             "frustum": frustum, "direccion": direccion, "cartas": cartas,
                             "trayectoria": puntos_tray.copy(),
                             "tramos_trayectoria": trayectoria.indices.copy()}
        if self._vis is not None:
            self._convertir_o3d(self._dinamica, self._g_dinamica)
            self._vis.update_geometry(self._g_dinamica)
            if not self._vis.poll_events():
                self.cerrar()
                return False
            self._vis.update_renderer()
        return True

    def renderizar_vista_externa(self, tamano=(960, 720)):
        """Vista ortográfica fija para PNG de diagnóstico, sin ventana/Open3D.

        No es una captura de la ventana interactiva: permite inspeccionar la
        geometría compartida cuando no hay pantalla ni Open3D instalado.
        """
        w, h = tamano
        if w < 320 or h < 240:
            raise ValueError("La vista de diagnóstico requiere al menos 320×240 píxeles")
        imagen = np.full((h, w, 3), (25, 22, 18), np.uint8)
        # Bases constantes: la posición estimada jamás modifica al observador.
        frente = np.array([-700., 950., -700.]); frente /= np.linalg.norm(frente)
        derecha = np.cross(frente, [0, 0, 1]); derecha /= np.linalg.norm(derecha)
        abajo = np.cross(frente, derecha)
        base = np.column_stack([derecha, abajo])
        escala = min(w, h) / 1550
        centro = np.array([0., 0., 320.])

        def proyectar(p):
            return (np.asarray(p)-centro) @ base * escala + [w*.5, h*.54]

        for geometria in (self._estatica, self._dinamica):
            pp = np.rint(proyectar(geometria.puntos)).astype(np.int32)
            for (a, b), rgb in zip(geometria.indices, geometria.colores):
                color = tuple(int(v) for v in np.rint(rgb[::-1]*255))
                cv2.line(imagen, tuple(pp[a]), tuple(pp[b]), color, 2, cv2.LINE_AA)
        textos = ["TP5 | Observador externo fijo | distancias en mm",
                  "X rojo: derecha | Y verde: fondo | Z azul: arriba",
                  "Frustum y direccion: +Z camara OpenCV | naranja punteada: ultima pose"]
        if self._estado is not None:
            p = self._estado.camara
            textos.append(f"Cuadro {p.frame_id} | " + ("CAMARA LOCALIZADA" if p.valida else "SEGUIMIENTO PERDIDO"))
            if p.valida:
                x, y, z = p.T_wc[:3, 3]
                textos.append(f"X {x:+.1f}  Y {y:+.1f}  Z {z:+.1f} mm | reproy {p.error_px or 0:.3f} px")
            elif p.antiguedad_s is not None:
                textos.append(f"Ultima pose: hace {p.antiguedad_s:.2f} s; no es una medicion actual")
        for i, texto in enumerate(textos):
            cv2.putText(imagen, texto, (16, 24+i*22), cv2.FONT_HERSHEY_SIMPLEX,
                        .49, (220, 225, 230), 1, cv2.LINE_AA)
        for i, nombre in enumerate(("+X [mm]", "+Y [mm]", "+Z [mm]")):
            p = np.zeros(3); p[i] = 160
            cv2.putText(imagen, nombre, tuple(np.rint(proyectar(p)).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1, cv2.LINE_AA)
        for id_marcador, esquinas in self.tablero.marcadores.items():
            p = np.rint(proyectar(esquinas.mean(axis=0))).astype(int)
            cv2.putText(imagen, str(id_marcador), tuple(p), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, (220, 220, 220), 1, cv2.LINE_AA)
        return imagen

    def cerrar(self):
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
        self._cerrado = True
