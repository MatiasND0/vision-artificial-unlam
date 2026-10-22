"""AR métrica sobre imagen cruda; proyección K/dist sin undistort previo.

El panel y las etiquetas de contraste adaptan el estilo de
``tp3-localizacion-homografica/localizacion/anotacion.py``. La geometría es nueva:
cada carta usa el plano XY de su marcador y crece hacia su +Z local.
"""
from collections.abc import Mapping
import unicodedata

import cv2
import numpy as np

from .modelos import Calibracion, CartaConfig, EstadoMundo


def _transformacion(T):
    T = np.asarray(T, dtype=np.float64)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6)):
        raise ValueError("La pose debe ser una transformación rígida finita 4×4")
    return T


def proyectar_puntos(puntos_mm, T_cm, calibracion: Calibracion):
    """Devuelve píxeles y máscara de puntos delante de la cámara.

    Las filas sin profundidad positiva quedan en NaN. No se aplica corrección
    a la imagen: OpenCV aplica una sola vez la distorsión de la calibración.
    """
    puntos = np.asarray(puntos_mm, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(puntos).all():
        raise ValueError("Los puntos 3D deben ser finitos y expresados en mm")
    T = _transformacion(T_cm)
    puntos_c = puntos @ T[:3, :3].T + T[:3, 3]
    visibles = puntos_c[:, 2] > 1e-6
    pixeles = np.full((len(puntos), 2), np.nan)
    if visibles.any():
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        proyectados, _ = cv2.projectPoints(
            puntos[visibles], rvec, T[:3, 3], calibracion.K, calibracion.dist)
        pixeles[visibles] = proyectados.reshape(-1, 2)
        visibles &= np.isfinite(pixeles).all(axis=1)
    return pixeles, visibles


def geometria_carta(lado_mm: float):
    """Base canónica ArUco y plano vertical simple, longitudes en mm."""
    if not np.isfinite(lado_mm) or lado_mm <= 0:
        raise ValueError("El lado del marcador debe ser positivo, en mm")
    l = float(lado_mm)
    return {
        "base": np.array([[-l/2, l/2, 0], [l/2, l/2, 0],
                          [l/2, -l/2, 0], [-l/2, -l/2, 0]]),
        "vertical": np.array([[-.7*l, 0, 0], [.7*l, 0, 0],
                              [.7*l, 0, 1.8*l], [-.7*l, 0, 1.8*l]]),
        "ejes": np.array([[0, 0, 0], [.7*l, 0, 0],
                          [0, .7*l, 0], [0, 0, .7*l]]),
    }


def geometria_cuadrado(lado_mm=40.0, altura_mm=20.0):
    """Cuadrado XY centrado en (0, 0, altura), en coordenadas del mundo y mm."""
    lado, altura = float(lado_mm), float(altura_mm)
    if not np.isfinite([lado, altura]).all() or lado <= 0 or altura < 0:
        raise ValueError("El lado debe ser positivo y la altura no negativa, finitos y en mm")
    mitad = lado / 2
    return np.array([[-mitad, mitad, altura], [mitad, mitad, altura],
                     [mitad, -mitad, altura], [-mitad, -mitad, altura]])


def _ascii(texto):
    # Hershey no contiene glifos Unicode; conserva nombres legibles.
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()


def _texto(imagen, texto, punto, color=(240, 240, 240), escala=.48):
    cv2.putText(imagen, _ascii(texto), punto, cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, 1, cv2.LINE_AA)


def _panel(imagen, lineas, color):
    """Franja de estado, adaptada del panel de TP3 (ver atribución arriba)."""
    alto = min(imagen.shape[0], 12 + 21 * len(lineas))
    imagen[:alto] = (imagen[:alto].astype(np.float32) * .3).astype(np.uint8)
    for i, linea in enumerate(lineas):
        _texto(imagen, linea, (10, 22 + 21*i), color)


def _segmento(imagen, a, b, color, grosor=2):
    if not np.isfinite([a, b]).all() or np.max(np.abs([a, b])) > 1e7:
        return
    alto, ancho = imagen.shape[:2]
    p, q = tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))
    dentro, p, q = cv2.clipLine((0, 0, ancho, alto), p, q)
    if dentro:
        cv2.line(imagen, p, q, color, grosor, cv2.LINE_AA)


def _contorno(imagen, puntos, visibles, color):
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        if visibles[a] and visibles[b]:
            _segmento(imagen, puntos[a], puntos[b], color)


def _etiqueta(imagen, lineas, origen, color):
    escala = .47
    ancho = max(cv2.getTextSize(_ascii(l), cv2.FONT_HERSHEY_SIMPLEX,
                                escala, 1)[0][0] for l in lineas) + 12
    alto = 19 * len(lineas) + 8
    ancho = min(ancho, imagen.shape[1] - 4)
    x = int(np.clip(origen[0], 2, max(2, imagen.shape[1] - ancho - 2)))
    y = int(np.clip(origen[1], 2, max(2, imagen.shape[0] - alto - 2)))
    cv2.rectangle(imagen, (x, y), (x+ancho, y+alto), (20, 20, 20), -1)
    cv2.rectangle(imagen, (x, y), (x+ancho, y+alto), color, 1)
    for i, linea in enumerate(lineas):
        _texto(imagen, linea, (x+6, y+18+19*i), escala=escala)


def dibujar_ar(imagen, calibracion: Calibracion, estado: EstadoMundo,
               cartas_config: Mapping[int, CartaConfig], *, cuadrado_ar=False):
    """Copia la imagen y anota solo observaciones de cartas del mismo cuadro.

    Una carta con T_cm actual puede proyectarse aunque se pierda el tablero:
    ese dibujo relativo a cámara no representa una localización en el mundo.
    El cuadrado opcional está fijo en el mundo y exige la pose actual T_cw.
    """
    calibracion.validar_imagen(imagen)
    if imagen.ndim != 3 or imagen.shape[2] != 3 or imagen.dtype != np.uint8:
        raise ValueError("AR requiere una imagen BGR uint8")
    salida = imagen.copy()
    camara = estado.camara
    dibujadas = 0
    for carta in estado.cartas:
        config = cartas_config.get(carta.id)
        if config is None or carta.frame_id != camara.frame_id:
            continue
        # No retener cartas: la única fuente es el EstadoMundo del cuadro actual.
        geometria = geometria_carta(config.lado_mm)
        base, vb = proyectar_puntos(geometria["base"], carta.T_cm, calibracion)
        plano, vp = proyectar_puntos(geometria["vertical"], carta.T_cm, calibracion)
        ejes, ve = proyectar_puntos(geometria["ejes"], carta.T_cm, calibracion)
        if not vb.all() or not vp.all():
            continue  # geometría cruzando el plano de cámara: no rellenar polígonos
        if (np.max(np.abs(base)) > 1e7 or np.max(np.abs(plano)) > 1e7):
            continue
        color = tuple(int(c) for c in config.color_bgr)
        capa = salida.copy()
        # Limitar primero evita desbordes de enteros ante poses casi tangenciales.
        poligono = np.rint(np.clip(plano, -100000, 100000)).astype(np.int32)
        cv2.fillConvexPoly(capa, poligono, color, cv2.LINE_AA)
        cv2.addWeighted(capa, .3, salida, .7, 0, salida)
        _contorno(salida, base, vb, color)
        _contorno(salida, plano, vp, color)
        _segmento(salida, plano[0], plano[2], color, 1)
        _segmento(salida, plano[1], plano[3], color, 1)
        for fin, color_eje in ((1, (40, 40, 255)), (2, (40, 230, 40)),
                               (3, (255, 100, 40))):
            if ve[0] and ve[fin]:
                _segmento(salida, ejes[0], ejes[fin], color_eje, 2)
        # Texto solo si el anclaje está visible; no pegar etiquetas al borde por
        # un marcador situado completamente fuera del encuadre.
        ancla = plano[2:4].mean(axis=0)
        if 0 <= ancla[0] < salida.shape[1] and 0 <= ancla[1] < salida.shape[0]:
            _etiqueta(salida, [config.nombre, f"ATK {config.ataque} / DEF {config.defensa}",
                              f"id {carta.id} | lado {config.lado_mm:g} mm | err {carta.error_px:.2f} px"],
                       ancla + [8, -55], color)
        dibujadas += 1

    if cuadrado_ar and camara.valida:
        cuadrado, visibles = proyectar_puntos(geometria_cuadrado(), camara.T_cw,
                                              calibracion)
        if visibles.all() and np.max(np.abs(cuadrado)) <= 1e7:
            color = (180, 240, 40)
            capa = salida.copy()
            poligono = np.rint(np.clip(cuadrado, -100000, 100000)).astype(np.int32)
            cv2.fillConvexPoly(capa, poligono, color, cv2.LINE_AA)
            cv2.addWeighted(capa, .25, salida, .75, 0, salida)
            _contorno(salida, cuadrado, visibles, color)

    estado_texto = "CAMARA LOCALIZADA" if camara.valida else "SEGUIMIENTO PERDIDO"
    if not camara.valida and estado.instancias_3d:
        estado_texto = "CAMARA SIN REFERENCIA FIJA | PERSONAJE VISIBLE"
    reproy = f"{camara.error_px:.3f} px" if camara.error_px is not None else "sin medicion"
    tamano = f"{calibracion.tamano[0]}x{calibracion.tamano[1]}"
    sintetica = (calibracion.metadatos.get("sintetica")
                 or calibracion.metadatos.get("origen") == "sintetica")
    origen = "SINTETICA" if sintetica else "archivo / parametros provistos"
    objetos = (f"personajes actuales {len(estado.instancias_3d)}" if estado.instancias_3d
               else f"cartas actuales {dibujadas}")
    lineas = [f"{estado_texto} | cuadro {camara.frame_id} | referencias {len(camara.ids_usados)} | reproy {reproy}",
              f"Calibracion {origen}: {tamano} | RMS {calibracion.rms_px:.3f} px | K/dist sobre imagen cruda",
              f"Mundo [mm]: X derecha, Y fondo, Z arriba | {objetos}"]
    if camara.valida:
        x, y, z = np.asarray(camara.T_wc)[:3, 3]
        lineas.append(f"Camara en mundo: X {x:+.1f}  Y {y:+.1f}  Z {z:+.1f} mm")
    else:
        motivo = camara.motivo or "sin referencias fijas validas"
        lineas.append(f"{motivo} | AR de carta relativa a camara; sin pose mundial actual")
    if cuadrado_ar:
        visibilidad = "pose actual" if camara.valida else "oculto: sin pose mundial actual"
        lineas.append(f"Cuadrado AR activo: 40x40 mm | altura +20 mm | {visibilidad}")
    _panel(salida, lineas, (100, 240, 100) if camara.valida else (60, 160, 255))
    return salida
