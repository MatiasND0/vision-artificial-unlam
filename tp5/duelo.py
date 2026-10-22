"""Combate visual determinista; no modifica las poses medidas por PnP.

Contrato de presentación: ResultadoDuelo.poses contiene poses SOLO de render;
estado es JSON con fase, progreso, pausado, motivo, vida_dragon, vida_mago,
ganador, tiempo_activo_s y ataque opcional. Un ataque contiene tipo
(carga/disparo/impacto), atacante (dragon/mago), progreso y origen_c/destino_c
en milímetros respecto de la cámara actual. Si falta un ancla no hay ataque.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .modelos import PoseCamara
from .pose import invertir_transformacion, transformar_puntos


# Combate visual simplificado; los daños no representan las reglas de las cartas.
FASES = (("preparacion", .6), ("dragon_carga", .8), ("dragon_disparo", .7),
         ("dragon_impacto", .45), ("mago_carga", .8), ("mago_disparo", .7),
         ("mago_impacto", .45), ("descanso", .6))
IDS = {"dragon": 0, "mago": 10}
MODELOS = {0: 0, 10: 1}


@dataclass
class ResultadoDuelo:
    poses: list
    estado: dict


def _traslacion(vector):
    T = np.eye(4)
    T[:3, 3] = vector
    return T


class ControlDuelo:
    """Fases por tiempo observado, daño único al impactar y pausa sin ambas anclas.

    Las poses de salida son cosméticas: miran al rival y animan pequeños avances
    y retrocesos. Las InstanciaModelo de entrada y las poses PnP no se modifican.
    """
    def __init__(self, modelos):
        if len(modelos) < 2:
            raise ValueError("El combate requiere los modelos de Dragón y Mago")
        self.perfiles = {}
        for id, indice in MODELOS.items():
            modelo = modelos[indice]
            principal = max(modelo.partes, key=lambda p: len(p.triangulos))
            v = np.asarray(principal.vertices_mm, dtype=float)
            if not len(v) or not np.isfinite(v).all():
                raise ValueError("El modelo de combate debe tener geometría finita")
            centro = np.median(v[:, :2], axis=0)
            z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
            pecho = z0 + .55*(z1-z0)
            boca = np.r_[centro + [0., -modelo.tamano_mm*(.34 if id == 0 else .16)],
                           pecho + (0. if id == 0 else .12*(z1-z0))]
            self.perfiles[id] = {"alinear": _traslacion([-centro[0], -centro[1], 0.]),
                                 "objetivo": np.r_[centro, pecho], "emisor": boca,
                                 "tamano_mm": float(modelo.tamano_mm)}
        self.reiniciar()

    def reiniciar(self):
        self.vidas = {"dragon": 100, "mago": 100}
        self.ganador = None
        self.victoria = False
        self.pausa_manual = False
        self._fase = 0
        self._en_fase = 0.
        self._tiempo_activo = 0.
        self._ultimo_tiempo = None
        self._presencia = 0.
        self._habilitado_anterior = False
        self._giros = {0: 0., 10: 0.}
        self.eventos = []

    def alternar_pausa(self):
        self.pausa_manual = not self.pausa_manual

    def _avanzar(self, dt):
        while dt > 1e-10 and not self.victoria:
            nombre, duracion = FASES[self._fase]
            paso = min(dt, duracion-self._en_fase)
            self._en_fase += paso
            self._tiempo_activo += paso
            dt -= paso
            if self._en_fase >= duracion-1e-10:
                if self.ganador is not None:
                    self.victoria = True
                    break
                self._fase = (self._fase+1) % len(FASES)
                self._en_fase = 0.
                nombre = FASES[self._fase][0]
                if nombre.endswith("_impacto"):
                    atacante = nombre.split("_")[0]
                    rival = "mago" if atacante == "dragon" else "dragon"
                    dano = 25 if atacante == "dragon" else 20
                    self.vidas[rival] = max(0, self.vidas[rival]-dano)
                    self.eventos.append({"atacante": atacante, "objetivo": rival,
                                         "dano": dano, "vida_restante": self.vidas[rival],
                                         "tiempo_activo_s": self._tiempo_activo})
                    if self.vidas[rival] == 0:
                        self.ganador = atacante

    def actualizar(self, instancias, frame_id, tiempo_s):
        tiempo_s = float(tiempo_s)
        if not np.isfinite(tiempo_s):
            raise ValueError("El tiempo del combate debe ser finito")
        dt = 0. if self._ultimo_tiempo is None else tiempo_s-self._ultimo_tiempo
        self._ultimo_tiempo = tiempo_s
        salto_tiempo = dt < 0 or dt > .5
        if salto_tiempo:
            self._presencia = 0.
            dt = 0.
        grupos = {id: [i for i in instancias if i.marcador_id == id and i.frame_id == frame_id]
                  for id in MODELOS}
        actuales = {}
        for id, grupo in grupos.items():
            if len(grupo) == 1:
                i = grupo[0]
                if i.modelo_indice != MODELOS[id]:
                    raise ValueError("El combate espera Dragón en ID 0 y Mago en ID 10")
                invertir_transformacion(i.T_cm)  # validar transformación rígida
                if i.T_cm[2, 3] > 1e-6:
                    actuales[id] = i
        ambos = len(actuales) == 2
        if ambos:
            separacion = np.linalg.norm(actuales[0].T_cm[:3, 3]-actuales[10].T_cm[:3, 3])
            ambos = bool(separacion >= 5.)
        habilitado = ambos and not self.pausa_manual
        dt_habilitado = max(0., dt) if habilitado and self._habilitado_anterior and not salto_tiempo else 0.
        self._habilitado_anterior = habilitado
        if not ambos:
            self._presencia = 0.
            motivo = "Mostrar ambos marcadores separados: ID 0 y ID 10"
        elif self.pausa_manual:
            self._presencia = 0.
            motivo = "Pausa manual | Espacio para continuar"
        else:
            anterior = self._presencia
            self._presencia = min(.25, self._presencia + dt_habilitado)
            activo_dt = max(0., dt_habilitado-max(0., .25-anterior))
            self._avanzar(activo_dt)
            motivo = "Preparando combate" if self._presencia < .25 else "Combate en curso"
        pausado = not ambos or self.pausa_manual or self._presencia < .25
        if self.victoria:
            motivo = "Gana Dragon Blanco" if self.ganador == "dragon" else "Gana Mago Oscuro"
        if ambos:
            for id, otro in ((0, 10), (10, 0)):
                T = actuales[id].T_cm
                direccion = T[:3, :3].T @ (actuales[otro].T_cm[:3, 3]-T[:3, 3])
                if np.linalg.norm(direccion[:2]) > 1e-6:
                    # Ambos assets miran hacia -Y local en su pose original.
                    self._giros[id] = float(np.arctan2(direccion[1], direccion[0])+np.pi/2)
        fase, duracion = FASES[self._fase]
        progreso = float(np.clip(self._en_fase/duracion, 0., 1.))
        atacante = fase.split("_")[0] if "_" in fase else None
        tipo = fase.split("_")[1] if "_" in fase else None
        poses = []
        transformaciones = {}
        for id, instancia in actuales.items():
            R = np.eye(4)
            R[:3, :3] = cv2.Rodrigues(np.array([0., 0., self._giros[id]]))[0]
            desplazamiento = np.zeros(3)
            inclinacion = np.eye(4)
            if not pausado and not self.victoria and atacante is not None:
                soy_atacante = id == IDS[atacante]
                if soy_atacante and tipo in ("carga", "disparo"):
                    amplitud = .045*self.perfiles[id]["tamano_mm"]
                    avance = np.sin(np.pi*progreso)*(.3 if tipo == "carga" else 1.)
                    desplazamiento = R[:3, :3] @ [0., -amplitud*avance, 0.]
                elif not soy_atacante and tipo == "impacto":
                    pulso = np.sin(np.pi*progreso)
                    inclinacion[:3, :3] = cv2.Rodrigues(np.array([np.radians(6)*pulso, 0., 0.]))[0]
                    desplazamiento = R[:3, :3] @ [0., .035*self.perfiles[id]["tamano_mm"]*pulso, 0.]
            T_render = (instancia.T_cm @ _traslacion(desplazamiento) @ R @ inclinacion
                        @ self.perfiles[id]["alinear"])
            transformaciones[id] = T_render
            poses.append((instancia.modelo_indice, PoseCamara(frame_id, tiempo_s, True,
                          T_render, invertir_transformacion(T_render), (id,), instancia.error_px)))
        ataque = None
        if not pausado and not self.victoria and atacante is not None:
            id_a = IDS[atacante]
            id_b = 10 if id_a == 0 else 0
            origen = transformar_puntos(transformaciones[id_a], self.perfiles[id_a]["emisor"])
            destino = transformar_puntos(transformaciones[id_b], self.perfiles[id_b]["objetivo"])
            ataque = {"tipo": tipo, "atacante": atacante, "progreso": progreso,
                      "origen_c": origen.tolist(), "destino_c": destino.tolist()}
        estado = {"fase": "victoria" if self.victoria else fase, "progreso": progreso,
                  "pausado": pausado, "motivo": motivo, "vida_dragon": self.vidas["dragon"],
                  "vida_mago": self.vidas["mago"], "ganador": self.ganador if self.victoria else None,
                  "tiempo_activo_s": self._tiempo_activo, "ataque": ataque,
                  "impactos": len(self.eventos)}
        return ResultadoDuelo(poses, estado)
