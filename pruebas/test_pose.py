"""Verdad sintética exacta separada del ensayo raster → detector → PnP."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tp5.configuracion import CARTAS
from tp5.deteccion import Detector
from tp5.modelos import Deteccion, PoseCamara, Tablero
from tp5.pose import EstimadorPose, estimar_cartas, invertir_transformacion, transformar_puntos
from tp5.simulador import Simulador, _mirar_a
from tp5.tablero import cargar_tablero, crear_tablero_ejemplo, esquinas_cuadrado, guardar_tablero


def error_angulo(T, verdad):
    delta = T[:3, :3].T @ verdad[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))


class PruebasPose(unittest.TestCase):
    def setUp(self):
        self.sim = Simulador()
        self.estimador = EstimadorPose(self.sim.tablero, self.sim.calibracion)

    def comprobar_pose(self, cuadro, detecciones, posicion_mm, angulo_grados):
        pose = self.estimador.estimar(detecciones, cuadro.frame_id, cuadro.tiempo_s)
        self.assertTrue(pose.valida, pose.motivo)
        self.assertLess(np.linalg.norm(pose.T_wc[:3, 3] - cuadro.T_wc[:3, 3]), posicion_mm)
        self.assertLess(error_angulo(pose.T_wc, cuadro.T_wc), angulo_grados)
        return pose

    def test_inversion_y_transformacion(self):
        cuadro = self.sim.generar(40)
        np.testing.assert_allclose(cuadro.T_cw @ invertir_transformacion(cuadro.T_cw), np.eye(4), atol=1e-10)
        puntos = np.array([[1, 2, 3], [-50, 20, 0.]])
        np.testing.assert_allclose(transformar_puntos(cuadro.T_wc, transformar_puntos(cuadro.T_cw, puntos)), puntos, atol=1e-10)
        with self.assertRaises(ValueError):
            invertir_transformacion(np.diag([2., 1, 1, 1]))

    def test_pose_exacta_con_camara_movil(self):
        for frame in [0, 30, 90, 180, 300, 550]:
            cuadro = self.sim.generar(frame)
            pose = self.comprobar_pose(cuadro, cuadro.detecciones, 1e-5, 1e-5)
            np.testing.assert_allclose(pose.T_cw, cuadro.T_cw, atol=1e-6)

    def test_cadena_raster_deteccion_pnp(self):
        # Tolerancias fijadas antes del ensayo: cámara 5 mm / 1°, carta 12 mm.
        detector = Detector()
        for frame in [0, 60, 180, 300]:
            cuadro = self.sim.generar(frame)
            detectadas = detector.detectar(cuadro.imagen)
            self.assertEqual([d.id for d in detectadas], [0, 1, 2, 3, 10, 11])
            pose = self.comprobar_pose(cuadro, detectadas, 5., 1.)
            cartas = estimar_cartas(detectadas, self.sim.calibracion, CARTAS, pose)
            self.assertEqual(len(cartas), 2)
            for carta in cartas:
                self.assertLess(np.linalg.norm(carta.T_wm[:3, 3] - cuadro.cartas_T_wm[carta.id][:3, 3]), 12.)

    def test_distorsion_cruda_exacta_y_render(self):
        self.sim.calibracion.dist[:] = [-.16, .03, .001, -.0005, 0.]
        cuadro = self.sim.generar(130)
        self.comprobar_pose(cuadro, cuadro.detecciones, 1e-4, 1e-4)
        detectadas = Detector().detectar(cuadro.imagen)
        self.assertEqual(len(detectadas), 6)
        self.comprobar_pose(cuadro, detectadas, 5., 1.)

    def test_independencia_mover_carta_camara_quieta(self):
        anterior = None
        cartas_anteriores = None
        for frame in [0, 45, 90]:
            cuadro = self.sim.generar(frame, mover_camara=False)
            pose = self.comprobar_pose(cuadro, cuadro.detecciones, 1e-5, 1e-5)
            cartas = estimar_cartas(cuadro.detecciones, self.sim.calibracion, CARTAS, pose)
            for carta in cartas:
                np.testing.assert_allclose(carta.T_wm, cuadro.cartas_T_wm[carta.id], atol=1e-5)
            if anterior is not None:
                np.testing.assert_allclose(pose.T_wc, anterior, atol=1e-6)
                self.assertGreater(np.linalg.norm(cartas[0].T_wm[:3, 3] - cartas_anteriores[0].T_wm[:3, 3]), 5.)
            anterior, cartas_anteriores = pose.T_wc, cartas

    def test_independencia_mover_camara_cartas_quietas(self):
        anterior = None
        for frame in [0, 60, 150]:
            cuadro = self.sim.generar(frame, mover_cartas=False)
            pose = self.comprobar_pose(cuadro, cuadro.detecciones, 1e-5, 1e-5)
            cartas = estimar_cartas(cuadro.detecciones, self.sim.calibracion, CARTAS, pose)
            for carta in cartas:
                np.testing.assert_allclose(carta.T_wm, cuadro.cartas_T_wm[carta.id], atol=1e-5)
            if anterior is not None:
                self.assertGreater(np.linalg.norm(pose.T_wc[:3, 3] - anterior[:3, 3]), 10.)
            anterior = pose.T_wc

    def test_perdida_solo_cartas_y_recuperacion(self):
        cuadro = self.sim.generar(0)
        valida = self.comprobar_pose(cuadro, cuadro.detecciones, 1e-5, 1e-5)
        oculto = self.sim.generar(30, ocultar_tablero=True)
        perdida = self.estimador.estimar(oculto.detecciones, 30, 1.)
        self.assertFalse(perdida.valida)
        self.assertIsNone(perdida.T_cw)
        self.assertIsNone(perdida.T_wc)
        self.assertAlmostEqual(perdida.antiguedad_s, 1.)
        np.testing.assert_allclose(perdida.ultima_T_wc, valida.T_wc)
        cartas = estimar_cartas(oculto.detecciones, self.sim.calibracion, CARTAS, perdida)
        self.assertEqual(len(cartas), 2)
        self.assertTrue(all(c.T_wm is None for c in cartas))
        recuperado = self.sim.generar(60)
        self.comprobar_pose(recuperado, recuperado.detecciones, 1e-5, 1e-5)

    def test_rechaza_esquina_corrupta_y_duplicados(self):
        cuadro = self.sim.generar(0)
        malas = [Deteccion(d.id, d.esquinas) for d in cuadro.detecciones]
        malas[0].esquinas[0] += [40, 30]
        pose = self.estimador.estimar(malas, 0, 0.)
        self.assertFalse(pose.valida)
        self.assertIn("Reproyección", pose.motivo)
        pose = self.estimador.estimar(cuadro.detecciones + [cuadro.detecciones[0]], 0, 0.)
        self.assertFalse(pose.valida)
        self.assertIn("duplicado", pose.motivo)

    def test_rechaza_camara_debajo_del_tablero(self):
        T_cw = _mirar_a(np.array([100., -460., -900.]), np.zeros(3))
        rvec = cv2.Rodrigues(T_cw[:3, :3])[0]
        detecciones = [Deteccion(i, cv2.projectPoints(puntos, rvec, T_cw[:3, 3],
                       self.sim.calibracion.K, self.sim.calibracion.dist)[0])
                       for i, puntos in self.sim.tablero.marcadores.items()]
        pose = self.estimador.estimar(detecciones, 0, 0.)
        self.assertFalse(pose.valida)
        self.assertIn("física", pose.motivo)

    def test_un_marcador_fijo_tambien_localiza(self):
        cuadro = self.sim.generar(40)
        solo_uno = [d for d in cuadro.detecciones if d.id == 1]
        self.comprobar_pose(cuadro, solo_uno, 1e-4, 1e-4)

    def test_diagnostico_reproduce_geometria_y_se_limpia_al_perder(self):
        # Un cuadrado pequeño y escorzado conserva sus dos ramas físicas.
        puntos = esquinas_cuadrado(32.)
        tablero = Tablero("4x4_50", {0: puntos}, (32., 32.))
        self.sim.calibracion.dist[:] = [-.16, .03, .001, -.0005, 0.]
        estimador = EstimadorPose(tablero, self.sim.calibracion)
        verdad = _mirar_a(np.array([400., 390., 335.]), np.zeros(3))
        esquinas = cv2.projectPoints(puntos, cv2.Rodrigues(verdad[:3, :3])[0],
                     verdad[:3, 3], self.sim.calibracion.K,
                     self.sim.calibracion.dist)[0].reshape(4, 2)
        pose = estimador.estimar([Deteccion(0, esquinas)], 148, 1.)
        self.assertTrue(pose.valida, pose.motivo)
        diagnostico = estimador.diagnostico
        # Debe poder persistirse sin conversiones especiales ni NaN/Infinity.
        json.dumps(diagnostico, allow_nan=False)
        self.assertEqual(diagnostico["frame_id"], 148)
        self.assertEqual(diagnostico["metodo"], "SOLVEPNP_IPPE")
        self.assertEqual(diagnostico["metodos_intentados"], ["SOLVEPNP_IPPE"])
        self.assertEqual(diagnostico["umbral_error_max_px"], 3.)
        self.assertEqual(diagnostico["margen_continuidad_px"], .05)
        self.assertGreaterEqual(len(diagnostico["candidatos"]), 2)
        self.assertEqual(diagnostico["seleccion"]["criterio"], "menor_rms")
        elegido = diagnostico["candidatos"][diagnostico["seleccion"]["indice"]]
        T_cw, T_wc = np.array(elegido["T_cw"]), np.array(elegido["T_wc"])
        np.testing.assert_allclose(T_cw, pose.T_cw, atol=1e-12)
        np.testing.assert_allclose(T_wc, pose.T_wc, atol=1e-12)
        np.testing.assert_allclose(T_cw @ T_wc, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(T_cw, verdad, atol=1e-5)
        proyectadas = cv2.projectPoints(puntos, cv2.Rodrigues(T_cw[:3, :3])[0],
                         T_cw[:3, 3], self.sim.calibracion.K,
                         self.sim.calibracion.dist)[0].reshape(4, 2)
        np.testing.assert_allclose(proyectadas, esquinas, atol=1e-6)
        residuos = np.linalg.norm(proyectadas - esquinas, axis=1)
        self.assertAlmostEqual(elegido["rms_px"], np.sqrt(np.mean(residuos ** 2)), places=9)
        self.assertAlmostEqual(elegido["error_max_px"], residuos.max(), places=9)
        self.assertAlmostEqual(elegido["rms_px"], pose.error_px, places=9)

        perdida = estimador.estimar([], 149, 1.1)
        self.assertFalse(perdida.valida)
        nuevo = estimador.diagnostico
        json.dumps(nuevo, allow_nan=False)
        self.assertIsNot(nuevo, diagnostico)
        self.assertEqual(nuevo["frame_id"], 149)
        self.assertIsNone(nuevo["metodo"])
        self.assertEqual(nuevo["metodos_intentados"], [])
        self.assertEqual(nuevo["candidatos"], [])
        self.assertIsNone(nuevo["seleccion"])
        # El registro previo sigue siendo una instantánea independiente.
        self.assertIsNotNone(diagnostico["seleccion"])
        np.testing.assert_allclose(perdida.ultima_T_wc, T_wc, atol=1e-12)

    def test_cartas_frontales_centradas_no_se_pierden(self):
        # Regresión OpenCV5: IPPE_SQUARE no entrega candidatos físicos para
        # estas observaciones válidas; SQPNP debe recuperar la misma pose.
        pose_perdida = PoseCamara(7, .2, False)
        puntos = esquinas_cuadrado(CARTAS[10].lado_mm)
        tvec = np.array([0., 0., 700.])
        for angulo in [np.pi, np.pi + .05]:
            with self.subTest(angulo=angulo):
                rvec = np.array([angulo, 0., 0.])
                esquinas = cv2.projectPoints(puntos, rvec, tvec,
                             self.sim.calibracion.K, self.sim.calibracion.dist)[0]
                cartas = estimar_cartas([Deteccion(10, esquinas)], self.sim.calibracion,
                                       CARTAS, pose_perdida)
                self.assertEqual(len(cartas), 1)
                carta = cartas[0]
                self.assertEqual(carta.frame_id, 7)
                self.assertIsNone(carta.T_wm)
                self.assertLess(carta.error_px, 1e-5)
                np.testing.assert_allclose(carta.T_cm[:3, :3], cv2.Rodrigues(rvec)[0], atol=1e-6)
                np.testing.assert_allclose(carta.T_cm[:3, 3], tvec, atol=1e-5)

    def test_tablero_frontal_centrado_no_se_pierde(self):
        tvec = np.array([0., 0., 700.])
        for angulo in [np.pi, np.pi + .05]:
            with self.subTest(angulo=angulo):
                rvec = np.array([angulo, 0., 0.])
                detecciones = [Deteccion(i, cv2.projectPoints(puntos, rvec, tvec,
                               self.sim.calibracion.K, self.sim.calibracion.dist)[0])
                               for i, puntos in self.sim.tablero.marcadores.items()]
                pose = self.estimador.estimar(detecciones, 8, .3)
                self.assertTrue(pose.valida, pose.motivo)
                self.assertLess(pose.error_px, 1e-4)
                np.testing.assert_allclose(pose.T_cw[:3, :3], cv2.Rodrigues(rvec)[0], atol=1e-6)
                np.testing.assert_allclose(pose.T_cw[:3, 3], tvec, atol=1e-4)

    def test_rescate_carta_no_acepta_dorso(self):
        puntos = esquinas_cuadrado(CARTAS[10].lado_mm)
        esquinas = cv2.projectPoints(puntos, np.zeros(3), np.array([0., 0., 700.]),
                     self.sim.calibracion.K, self.sim.calibracion.dist)[0]
        cartas = estimar_cartas([Deteccion(10, esquinas)], self.sim.calibracion,
                               CARTAS, PoseCamara(9, .4, False))
        self.assertEqual(cartas, [])

    def test_tablero_json_mm(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "tablero.json"
            original = crear_tablero_ejemplo()
            guardar_tablero(original, ruta)
            cargado = cargar_tablero(ruta)
            self.assertTrue(cargado.es_ejemplo)
            self.assertEqual(set(cargado.marcadores), {0, 1, 2, 3})
            for i in original.marcadores:
                np.testing.assert_array_equal(original.marcadores[i], cargado.marcadores[i])
            datos = json.loads(ruta.read_text())
            datos["unidades"] = "m"
            ruta.write_text(json.dumps(datos))
            with self.assertRaises(ValueError):
                cargar_tablero(ruta)


if __name__ == "__main__":
    unittest.main()
