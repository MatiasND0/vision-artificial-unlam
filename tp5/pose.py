"""Localización PnP por cuadro; T_ab lleva coordenadas de b a a, en mm."""
import cv2
import numpy as np

from .modelos import PoseCamara, PoseCarta
from .tablero import esquinas_cuadrado


def invertir_transformacion(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("Se requiere una transformación homogénea finita 4×4")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError("La rotación debe pertenecer a SO(3)")
    inversa = np.eye(4)
    inversa[:3, :3] = R.T
    inversa[:3, 3] = -R.T @ T[:3, 3]
    return inversa


def transformar_puntos(T, puntos):
    puntos = np.asarray(puntos, dtype=np.float64)
    if puntos.shape[-1] != 3:
        raise ValueError("Los puntos deben tener tres coordenadas")
    T = np.asarray(T, dtype=np.float64)
    return puntos @ T[:3, :3].T + T[:3, 3]


def _transformacion(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


def _resolver(objeto, imagen, calibracion, metodo, refinar=True):
    objeto = np.ascontiguousarray(objeto, dtype=np.float64)
    imagen = np.ascontiguousarray(imagen, dtype=np.float64)
    try:
        resultado = cv2.solvePnPGeneric(objeto, imagen, calibracion.K, calibracion.dist,
                                       flags=metodo)
    except cv2.error:
        return []
    candidatos = []
    if not resultado[0]:
        return candidatos
    for rvec, tvec in zip(resultado[1], resultado[2]):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        # Refinar cada solución conserva ambas ramas hasta aplicar los filtros.
        if refinar:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(objeto, imagen, calibracion.K,
                                                 calibracion.dist, rvec.copy(), tvec.copy())
            except cv2.error:
                continue
        T = _transformacion(rvec, tvec)
        if not np.isfinite(T).all() or np.any(transformar_puntos(T, objeto)[:, 2] <= 1e-6):
            continue
        proyeccion = cv2.projectPoints(objeto, rvec, tvec, calibracion.K, calibracion.dist)[0].reshape(-1, 2)
        residuos = np.linalg.norm(proyeccion - imagen, axis=1)
        candidatos.append((T, float(np.sqrt(np.mean(residuos ** 2))), float(residuos.max())))
    return candidatos


class EstimadorPose:
    def __init__(self, tablero, calibracion, error_max_px=3.0, *, verificar_continuidad=False):
        if not np.isfinite(error_max_px) or error_max_px <= 0:
            raise ValueError("El límite de reproyección debe ser positivo")
        self.tablero = tablero
        self.calibracion = calibracion
        self.error_max_px = float(error_max_px)
        self.verificar_continuidad = bool(verificar_continuidad)
        self._ultima_T_wc = None
        self._ultimo_tiempo = None
        self.diagnostico = self._nuevo_diagnostico(None)

    def _nuevo_diagnostico(self, frame_id):
        """Observabilidad JSON del cuadro, independiente de la última pose válida."""
        return {"frame_id": None if frame_id is None else int(frame_id),
                "metodo": None, "metodos_intentados": [],
                "umbral_error_max_px": self.error_max_px,
                "margen_continuidad_px": 0.05,
                "candidatos": [], "seleccion": None}

    def _perdida(self, frame_id, tiempo_s, motivo, ids=(), error=None):
        return PoseCamara(frame_id, tiempo_s, False, ids_usados=ids, error_px=error,
                          motivo=motivo,
                          ultima_T_wc=None if self._ultima_T_wc is None else self._ultima_T_wc.copy(),
                          antiguedad_s=None if self._ultimo_tiempo is None else max(0., tiempo_s - self._ultimo_tiempo))

    def estimar(self, detecciones, frame_id, tiempo_s):
        self.diagnostico = self._nuevo_diagnostico(frame_id)
        observadas = [d for d in detecciones if d.id in self.tablero.marcadores]
        ids = tuple(d.id for d in observadas)
        if not observadas:
            return self._perdida(frame_id, tiempo_s, "Sin referencias fijas visibles")
        if len(set(ids)) != len(ids):
            return self._perdida(frame_id, tiempo_s, "ID fijo duplicado en el cuadro", ids)
        objeto = np.concatenate([self.tablero.marcadores[d.id] for d in observadas])
        imagen = np.concatenate([d.esquinas for d in observadas])
        centrados = objeto - objeto.mean(axis=0)
        rango = np.linalg.matrix_rank(centrados, tol=1e-6)
        if rango < 2:
            return self._perdida(frame_id, tiempo_s, "Referencias degeneradas", ids)
        metodo = cv2.SOLVEPNP_IPPE if rango == 2 else cv2.SOLVEPNP_SQPNP
        nombre_metodo = "SOLVEPNP_IPPE" if rango == 2 else "SOLVEPNP_SQPNP"
        self.diagnostico["metodo"] = nombre_metodo
        self.diagnostico["metodos_intentados"].append(nombre_metodo)
        # El tablero se observa por arriba: la cámara debe estar en Z mundo > 0.
        def fisicos(candidatos):
            return [(T, error, maximo) for T, error, maximo in candidatos
                    if invertir_transformacion(T)[2, 3] > 1e-6]
        candidatos = fisicos(_resolver(objeto, imagen, self.calibracion, metodo))
        metodos_candidatos = {id(c): nombre_metodo for c in candidatos}
        if metodo == cv2.SOLVEPNP_IPPE and not any(c[2] <= self.error_max_px for c in candidatos):
            # IPPE de OpenCV puede degenerar con vistas frontales centradas.
            # SQPNP solo rescata la observación si supera los mismos filtros.
            self.diagnostico["metodos_intentados"].append("SOLVEPNP_SQPNP")
            rescate = fisicos(_resolver(objeto, imagen, self.calibracion,
                                       cv2.SOLVEPNP_SQPNP))
            metodos_candidatos.update({id(c): "SOLVEPNP_SQPNP" for c in rescate})
            candidatos.extend(rescate)
        if not candidatos:
            return self._perdida(frame_id, tiempo_s, "PnP sin solución física válida", ids)
        candidatos.sort(key=lambda c: c[1])
        self.diagnostico["candidatos"] = [
            {"metodo": metodos_candidatos[id(c)],
             "T_cw": c[0].tolist(),
             "T_wc": invertir_transformacion(c[0]).tolist(),
             "rms_px": c[1], "error_max_px": c[2]}
            for c in candidatos]
        aceptables = [c for c in candidatos if c[2] <= self.error_max_px]
        if not aceptables:
            return self._perdida(frame_id, tiempo_s, "Reproyección excede el límite por esquina", ids, candidatos[0][1])
        dt = None if self._ultimo_tiempo is None else tiempo_s - self._ultimo_tiempo
        if (self.verificar_continuidad and self._ultima_T_wc is not None
                and dt is not None and 0 < dt <= .5):
            anterior = invertir_transformacion(self._ultima_T_wc)
            limite_giro = 10. + 120. * dt
            limite_distancia = 25. + 1200. * dt
            continuas, rechazados = [], []
            for c in aceptables:
                R = anterior[:3, :3].T @ c[0][:3, :3]
                giro = float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))
                distancia = float(np.linalg.norm(c[0][:3, 3] - anterior[:3, 3]))
                if giro <= limite_giro and distancia <= limite_distancia:
                    continuas.append(c)
                else:
                    rechazados.append({"indice": next(i for i, v in enumerate(candidatos) if v is c),
                                        "giro_grados": giro, "distancia_mm": distancia})
            self.diagnostico["continuidad"] = {"delta_t_s": float(dt),
                "limite_giro_grados": limite_giro, "limite_distancia_mm": limite_distancia,
                "rechazados": rechazados}
            aceptables = continuas
            if not aceptables:
                return self._perdida(frame_id, tiempo_s, "Salto de pose sin confirmar", ids)
        mejor_error = aceptables[0][1]
        cercanas = [c for c in aceptables if c[1] <= mejor_error + 0.05]
        criterio = "menor_rms"
        if self._ultima_T_wc is not None and len(cercanas) > 1:
            def distancia(c):
                actual = invertir_transformacion(c[0])
                delta_R = self._ultima_T_wc[:3, :3].T @ actual[:3, :3]
                angulo = np.arccos(np.clip((np.trace(delta_R) - 1) / 2, -1, 1))
                return np.linalg.norm(actual[:3, 3] - self._ultima_T_wc[:3, 3]) + 100 * angulo
            elegida = min(cercanas, key=distancia)
            criterio = "continuidad_temporal"
        else:
            elegida = aceptables[0]
        self.diagnostico["seleccion"] = {
            "indice": next(i for i, c in enumerate(candidatos) if c is elegida),
            "criterio": criterio}
        T_cw, error, _ = elegida
        T_wc = invertir_transformacion(T_cw)
        self._ultima_T_wc = T_wc.copy()
        self._ultimo_tiempo = tiempo_s
        return PoseCamara(frame_id, tiempo_s, True, T_cw, T_wc, ids, error,
                          "Referencias fijas actuales", T_wc.copy(), 0.0)


def estimar_cartas(detecciones, calibracion, cartas_config, pose_camara):
    """IPPE_SQUARE local, con rescate SQPNP si no deja soluciones válidas.

    Las detecciones y pose_camara pertenecen al frame_id indicado por el llamador.
    Si falta tablero aún se produce T_cm para AR relativa a la cámara.
    Ambas vías exigen profundidad positiva, normal frontal y reproyección ≤3 px.
    """
    configs = cartas_config if isinstance(cartas_config, dict) else {c.id: c for c in cartas_config}
    salida = []
    repetidos = {d.id for d in detecciones if sum(e.id == d.id for e in detecciones) > 1}
    for deteccion in detecciones:
        if deteccion.id not in configs or deteccion.id in repetidos:
            continue
        puntos = esquinas_cuadrado(configs[deteccion.id].lado_mm)
        candidatos = []
        for metodo in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_SQPNP):
            candidatos = _resolver(puntos, deteccion.esquinas, calibracion, metodo)
            # Normal +Z debe apuntar hacia la cámara que ve el frente.
            candidatos = [c for c in candidatos
                          if np.dot(c[0][:3, 2], c[0][:3, 3]) < 0 and c[2] <= 3.0]
            if candidatos:
                break
        if not candidatos:
            continue
        T_cm, error, _ = min(candidatos, key=lambda c: c[1])
        T_wm = pose_camara.T_wc @ T_cm if pose_camara.valida else None
        salida.append(PoseCarta(deteccion.id, pose_camara.frame_id, T_cm, error, T_wm))
    return salida
