"""Anclas fijas y móviles de modelos, sin usar una carta para localizar cámara."""
import numpy as np

from .modelos import InstanciaModelo, PoseCamara
from .pose import invertir_transformacion


def marco_marcador(esquinas):
    puntos = np.asarray(esquinas, dtype=float).reshape(4, 3)
    x = puntos[1] - puntos[0]
    y = puntos[0] - puntos[3]
    z = np.cross(x, y)
    if not np.isfinite(puntos).all() or np.linalg.norm(x) < 1e-6 or np.linalg.norm(z) < 1e-6:
        raise ValueError("El marcador fijo debe definir un plano no degenerado")
    x /= np.linalg.norm(x)
    z /= np.linalg.norm(z)
    T_wm = np.eye(4)
    T_wm[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T_wm[:3, 3] = puntos.mean(axis=0)
    return T_wm


def crear_instancias(estado, tablero, modelos_por_marcador):
    """Cada T_cm corresponde al cuadro actual; T_wm solo existe con mundo válido."""
    instancias = []
    for identificador, indice in modelos_por_marcador.items():
        if identificador in tablero.marcadores:
            if not estado.camara.valida:
                continue
            T_wm = marco_marcador(tablero.marcadores[identificador])
            instancias.append(InstanciaModelo(identificador, indice, estado.camara.frame_id,
                              estado.camara.T_cw @ T_wm, estado.camara.error_px, T_wm))
        else:
            cartas = [c for c in estado.cartas if c.id == identificador
                      and c.frame_id == estado.camara.frame_id]
            if len(cartas) == 1:
                carta = cartas[0]
                instancias.append(InstanciaModelo(identificador, indice, carta.frame_id,
                                  carta.T_cm, carta.error_px, carta.T_wm))
    return instancias


def poses_para_render(instancias, tiempo_s):
    # Aquí el sistema local es el marcador. Estas inversas no reemplazan la
    # verdadera pose de cámara en EstadoMundo ni crean una localización mundial.
    return [(i.modelo_indice, PoseCamara(i.frame_id, tiempo_s, True, i.T_cm,
             invertir_transformacion(i.T_cm), (i.marcador_id,), i.error_px)) for i in instancias]
