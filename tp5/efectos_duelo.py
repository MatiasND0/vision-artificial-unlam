"""Presentación de combate visual: energía en cámara/mm y HUD, sin reglas de juego.

El motor entrega origen/destino en la cámara ACTUAL. Aquí solo se proyectan con
identidad y K/dist; no se conserva una pose ni se corrige la imagen de la cámara.
Los efectos son deterministas para un mismo estado de duelo.
"""
import cv2
import numpy as np

from .ar import _ascii, _segmento, proyectar_puntos


_CIAN = (255, 210, 65)
_VIOLETA = (255, 85, 210)
_IDENTIDAD = np.eye(4)


def _validar_imagen(imagen):
    if imagen.ndim != 3 or imagen.shape[2] != 3 or imagen.dtype != np.uint8:
        raise ValueError("Los efectos de duelo requieren imagen BGR uint8")


def _observacion_ataque(estado):
    ataque = estado.get("ataque")
    if estado.get("pausado", False) or ataque is None:
        return None
    if ataque.get("tipo") not in ("carga", "disparo", "impacto"):
        raise ValueError("Tipo de ataque desconocido")
    if ataque.get("atacante") not in ("dragon", "mago"):
        raise ValueError("Atacante desconocido")
    origen = np.asarray(ataque["origen_c"], dtype=float).reshape(3)
    destino = np.asarray(ataque["destino_c"], dtype=float).reshape(3)
    progreso = float(ataque["progreso"])
    if (not np.isfinite([origen, destino]).all() or not np.isfinite(progreso)
            or origen[2] <= 1e-6 or destino[2] <= 1e-6):
        return None
    return ataque["tipo"], ataque["atacante"], float(np.clip(progreso, 0, 1)), origen, destino


def _pixeles(puntos, calibracion):
    pixeles, visibles = proyectar_puntos(puntos, _IDENTIDAD, calibracion)
    visibles &= np.isfinite(pixeles).all(axis=1)
    visibles &= np.max(np.abs(pixeles), axis=1) < 1e7
    return pixeles, visibles


def _esfera(capa, punto, radio_mm, calibracion, color, brillo=1., nucleo=False):
    pixeles, visibles = _pixeles([punto, punto + [radio_mm, 0, 0]], calibracion)
    if not visibles.all() or brillo <= 0:
        return
    centro = pixeles[0]
    radio = float(np.linalg.norm(pixeles[1]-centro))
    if not np.isfinite(radio) or radio < .15:
        return
    radio = int(np.clip(np.rint(radio), 1, 40))
    alto, ancho = capa.shape[:2]
    if not -radio <= centro[0] < ancho+radio or not -radio <= centro[1] < alto+radio:
        return
    centro = tuple(np.rint(centro).astype(int))
    tono = tuple(int(np.clip(c*brillo, 0, 255)) for c in color)
    cv2.circle(capa, centro, radio, tono, -1, cv2.LINE_AA)
    if nucleo:
        blanco = int(np.clip(255*brillo, 0, 255))
        cv2.circle(capa, centro, max(1, radio//3), (blanco, blanco, blanco), -1, cv2.LINE_AA)


def _trazo(capa, puntos, calibracion, color, grosor=1):
    pixeles, visibles = _pixeles(puntos, calibracion)
    for i in range(len(pixeles)-1):
        if visibles[i] and visibles[i+1]:
            _segmento(capa, pixeles[i], pixeles[i+1], color, grosor)


def _carga(capa, origen, progreso, calibracion, color):
    radio = 1.5 + 4*progreso
    for i in range(8):
        a = i*np.pi/4 + 7*progreso
        r = radio*(1.8-.6*progreso)
        punto = origen + [r*np.cos(a), r*np.sin(a), 0]
        _esfera(capa, punto, .65, calibracion, color, .5+.4*progreso)
    _esfera(capa, origen, radio, calibracion, color, nucleo=True)


def _disparo(capa, origen, destino, progreso, calibracion, atacante, color):
    direccion = destino-origen
    if atacante == "dragon":
        # Frente del rayo que avanza en 3D; la cola recorre la misma trayectoria.
        u = np.linspace(max(0., progreso-.55), progreso, 25)
        puntos = origen + u[:, None]*direccion
        _trazo(capa, puntos, calibracion, color, 3)
        _trazo(capa, puntos, calibracion, (225, 245, 255), 1)
        desplazamiento = np.column_stack([np.zeros(len(u)),
            1.4*np.sin(u*45 + progreso*15)*np.sin(np.linspace(0, np.pi, len(u))),
            np.zeros(len(u))])
        _trazo(capa, puntos + desplazamiento, calibracion, color, 1)
        _esfera(capa, puntos[-1], 2.8, calibracion, color, nucleo=True)
    else:
        # Orbe violeta con arco suave, medido en mm de la cámara actual.
        u = np.linspace(max(0., progreso-.20), progreso, 10)
        puntos = origen + u[:, None]*direccion
        puntos[:, 1] -= 8*np.sin(np.pi*u)
        for i, punto in enumerate(puntos[:-1]):
            intensidad = (i+1)/len(puntos)
            _esfera(capa, punto, .7+2.0*intensidad, calibracion, color, .65*intensidad)
        _esfera(capa, puntos[-1], 3.8+.6*np.sin(np.pi*progreso), calibracion, color, nucleo=True)


def _impacto(capa, destino, progreso, calibracion, color):
    brillo = (1-progreso)**1.2
    radio = 3 + 18*progreso
    for i in range(12):
        a = i*2*np.pi/12 + .7*progreso
        r = radio*(.55+.45*(i % 3)/2)
        vector = np.array([r*np.cos(a), r*np.sin(a), (i % 3-1)*radio*.18])
        tono = tuple(int(c*brillo) for c in color)
        _trazo(capa, [destino+.45*vector, destino+vector], calibracion, tono, 1)
        _esfera(capa, destino+vector, .8, calibracion, color, brillo)
    _esfera(capa, destino, (1-progreso)*5+.5, calibracion, color, brillo, nucleo=True)


def dibujar_ataques(imagen, calibracion, estado_duelo, *, frente=False):
    """Copia BGR: carga/disparo al fondo; únicamente impacto en primer plano.

    Sin ataque, en pausa o con referencias inválidas se devuelve la imagen sin
    overlay. Los halos son sprites luminosos, no geometría con profundidad GPU.
    """
    calibracion.validar_imagen(imagen)
    _validar_imagen(imagen)
    observacion = _observacion_ataque(estado_duelo)
    if observacion is None:
        return imagen.copy()
    tipo, atacante, progreso, origen, destino = observacion
    if bool(frente) != (tipo == "impacto"):
        return imagen.copy()
    capa = np.zeros_like(imagen)
    color = _CIAN if atacante == "dragon" else _VIOLETA
    if tipo == "carga":
        _carga(capa, origen, progreso, calibracion, color)
    elif tipo == "disparo":
        _disparo(capa, origen, destino, progreso, calibracion, atacante, color)
    else:
        _impacto(capa, destino, progreso, calibracion, color)
    glow = cv2.GaussianBlur(capa, (0, 0), 2.2)
    return np.uint8(np.clip(imagen.astype(np.float32) + .7*capa + glow, 0, 255))


def _texto(imagen, texto, x, y, ancho, escala=.42, color=(240, 240, 245)):
    texto = _ascii(texto)
    if ancho <= 0:
        return
    original = texto
    while texto and cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0] > ancho:
        original = original[:-1]
        texto = original + "..." if original else ""
    if texto:
        cv2.putText(imagen, texto, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, color, 1, cv2.LINE_AA)


def _vida(valor):
    valor = float(valor)
    return int(np.clip(valor, 0, 100)) if np.isfinite(valor) else 0


def dibujar_hud(imagen, estado_duelo):
    """Dos barras y estado en y=105..175; victoria moderada solo al terminar."""
    _validar_imagen(imagen)
    salida = imagen.copy()
    alto, ancho = salida.shape[:2]
    if alto < 24 or ancho < 48:
        return salida
    arriba = min(105, max(0, alto-74))
    abajo = min(alto, arriba+70)
    margen = min(16, max(6, ancho//40))
    zona = salida[arriba:abajo, margen:ancho-margen]
    fondo = np.full_like(zona, (15, 18, 26))
    cv2.addWeighted(zona, .28, fondo, .72, 0, zona)
    escala = .42 if ancho < 640 else .48
    separacion = max(10, ancho//70)
    barra_ancho = (ancho-2*margen-separacion)//2
    vidas = (_vida(estado_duelo.get("vida_dragon", 100)),
             _vida(estado_duelo.get("vida_mago", 100)))
    for i, (nombre, vida, color) in enumerate(zip(("DRAGON", "MAGO"), vidas, (_CIAN, _VIOLETA))):
        x = margen + i*(barra_ancho+separacion)
        _texto(salida, f"{nombre}  {vida}/100", x+3, arriba+14, barra_ancho-6, escala)
        y1, y2 = arriba+20, arriba+30
        cv2.rectangle(salida, (x, y1), (x+barra_ancho, y2), (42, 45, 54), -1)
        if vida:
            cv2.rectangle(salida, (x, y1), (x+int(round(barra_ancho*vida/100)), y2), color, -1)
        cv2.rectangle(salida, (x, y1), (x+barra_ancho, y2), (140, 145, 155), 1)
    ganador = estado_duelo.get("ganador")
    motivo = str(estado_duelo.get("motivo", "") or "")
    fase = str(estado_duelo.get("fase", "esperando")).replace("_", " ").upper()
    if ganador in ("dragon", "mago"):
        estado = f"VICTORIA: {ganador.upper()} | R para reiniciar"
    elif estado_duelo.get("pausado", False):
        estado = "PAUSA: " + (motivo or "pausa manual")
    else:
        estado = f"FASE: {fase}" + (" | " + motivo if motivo else "")
    _texto(salida, estado, margen+3, arriba+47, ancho-2*margen-6, escala)
    _texto(salida, "Espacio: pausa | R: reinicia | Q: salir", margen+3, arriba+63,
           ancho-2*margen-6, .38 if ancho < 640 else .43, (205, 210, 220))
    if ganador in ("dragon", "mago") and alto-abajo >= 45:
        cartel_ancho = min(420, int(ancho*.65))
        cartel_alto = 34 if alto < 400 else 44
        x = (ancho-cartel_ancho)//2
        y = min(max(abajo+10, int(alto*.38)), alto-cartel_alto-8)
        color = _CIAN if ganador == "dragon" else _VIOLETA
        zona = salida[y:y+cartel_alto, x:x+cartel_ancho]
        cv2.addWeighted(zona, .3, np.full_like(zona, (18, 20, 32)), .7, 0, zona)
        cv2.rectangle(salida, (x, y), (x+cartel_ancho, y+cartel_alto), color, 1)
        _texto(salida, f"VICTORIA: {ganador.upper()}", x+12, y+cartel_alto//2+5,
               cartel_ancho-24, .48 if ancho < 640 else .65, color)
    return salida
