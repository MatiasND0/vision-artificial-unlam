"""Pruebas del pipeline de imagen, distintas de las pruebas de esquinas exactas."""
from contextlib import ExitStack, redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from main import ejecutar, error_transformacion, parser_cli, procesar_cuadro
from tp5.ar import geometria_cuadrado
from tp5.calibracion import guardar_calibracion
from tp5.configuracion import CARTAS
from tp5.deteccion import Detector
from tp5.modelos import Calibracion, Tablero
from tp5.pose import EstimadorPose
from tp5.simulador import Simulador
from tp5.tablero import esquinas_cuadrado, guardar_tablero
from tp5.visualizador import Visualizador3D


class IntegracionTest(unittest.TestCase):
    def setUp(self):
        self.sim = Simulador()
        self.detector = Detector(self.sim.tablero.diccionario)
        self.estimador = EstimadorPose(self.sim.tablero, self.sim.calibracion)

    def procesar(self, frame_id, **kwargs):
        cuadro = self.sim.generar(frame_id, **kwargs)
        original = cuadro.imagen.copy()
        estado, ar = procesar_cuadro(cuadro.imagen, self.sim.calibracion, self.detector,
                                    self.estimador, cuadro.frame_id, cuadro.tiempo_s)
        np.testing.assert_array_equal(cuadro.imagen, original)
        self.assertEqual(ar.shape, original.shape)
        self.assertGreater(np.count_nonzero(ar != original), 100)
        self.assertTrue(all(c.frame_id == frame_id for c in estado.cartas))
        return cuadro, estado

    def test_imagen_a_mundo_y_vista_externa(self):
        visor = Visualizador3D(self.sim.tablero, self.sim.calibracion, habilitado=False)
        try:
            for frame_id in (0, 60, 120):
                cuadro, estado = self.procesar(frame_id)
                self.assertTrue(estado.camara.valida, estado.camara.motivo)
                ep, ea = error_transformacion(estado.camara.T_wc, cuadro.T_wc)
                self.assertLess(ep, 5.)
                self.assertLess(ea, 1.)
                self.assertEqual({c.id for c in estado.cartas}, set(CARTAS))
                for c in estado.cartas:
                    ep, _ = error_transformacion(c.T_wm, cuadro.cartas_T_wm[c.id])
                    self.assertLess(ep, 12.)
                self.assertTrue(visor.actualizar(estado))
            vista = visor.renderizar_vista_externa()
            self.assertEqual(vista.ndim, 3)
            self.assertGreater(float(np.std(vista)), 5.)
        finally:
            visor.cerrar()

    def test_perdida_no_inventa_mundo_y_recupera(self):
        _, inicial = self.procesar(0)
        self.assertTrue(inicial.camara.valida)
        _, perdida = self.procesar(1, ocultar_tablero=True)
        self.assertFalse(perdida.camara.valida)
        self.assertIsNone(perdida.camara.T_wc)
        self.assertIsNone(perdida.camara.T_cw)
        self.assertIsNotNone(perdida.camara.ultima_T_wc)
        self.assertGreater(perdida.camara.antiguedad_s, 0)
        self.assertTrue(perdida.cartas)
        self.assertTrue(all(c.T_wm is None for c in perdida.cartas))
        _, recuperada = self.procesar(2)
        self.assertTrue(recuperada.camara.valida)
        self.assertTrue(all(c.T_wm is not None for c in recuperada.cartas))

    def test_distorsion_en_imagen_cruda(self):
        self.sim.calibracion.dist[:] = [-.14, .03, .001, -.0007, .002]
        cuadro, estado = self.procesar(50)
        self.assertTrue(estado.camara.valida, estado.camara.motivo)
        ep, ea = error_transformacion(estado.camara.T_wc, cuadro.T_wc)
        self.assertLess(ep, 5.)
        self.assertLess(ea, 1.)
        for c in estado.cartas:
            ep, _ = error_transformacion(c.T_wm, cuadro.cartas_T_wm[c.id])
            self.assertLess(ep, 12.)

    def test_rechaza_resolucion_distinta_antes_de_pnp(self):
        imagen = np.zeros((360, 640, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Resolución"):
            procesar_cuadro(imagen, self.sim.calibracion, self.detector, self.estimador, 0, 0.)

    def test_cuadrado_anclado_a_mundo_se_oculta_y_recupera(self):
        referencia = EstimadorPose(self.sim.tablero, self.sim.calibracion)
        proyecciones = []
        for frame_id, ocultar in ((0, False), (1, True), (60, False)):
            cuadro = self.sim.generar(frame_id, ocultar_tablero=ocultar)
            original = cuadro.imagen.copy()
            estado, con_cuadrado = procesar_cuadro(
                cuadro.imagen, self.sim.calibracion, self.detector, self.estimador,
                frame_id, cuadro.tiempo_s, {}, cuadrado_ar=True)
            sin_estado, sin_cuadrado = procesar_cuadro(
                cuadro.imagen, self.sim.calibracion, self.detector, referencia,
                frame_id, cuadro.tiempo_s, {})
            np.testing.assert_array_equal(cuadro.imagen, original)
            diferencia = np.any(con_cuadrado != sin_cuadrado, axis=2)
            diferencia[:160] = False  # El panel también indica el modo AR activo.
            self.assertEqual(estado.camara.valida, not ocultar)
            if ocultar:
                self.assertIsNotNone(estado.camara.ultima_T_wc)
                self.assertFalse(diferencia.any())
                continue
            np.testing.assert_allclose(estado.camara.T_cw, sin_estado.camara.T_cw)
            self.assertGreater(np.count_nonzero(diferencia), 100)
            # Usar la cámara verdadera del raster para comprobar el anclaje.
            T_cw = np.linalg.inv(cuadro.T_wc)
            pixeles, _ = cv2.projectPoints(
                geometria_cuadrado(), cv2.Rodrigues(T_cw[:3, :3])[0],
                T_cw[:3, 3], self.sim.calibracion.K, self.sim.calibracion.dist)
            pixeles = pixeles.reshape(-1, 2)
            for x, y in np.rint(pixeles).astype(int):
                self.assertTrue(diferencia[y-3:y+4, x-3:x+4].any())
            proyecciones.append(pixeles)
        self.assertGreater(np.linalg.norm(proyecciones[0] - proyecciones[1]), 5.)

    def test_verificador_rechaza_cartas_que_desaparecen(self):
        detectar_original = Detector.detectar
        llamadas = 0

        def detectar_sin_cartas_desde_segundo_cuadro(detector, imagen):
            nonlocal llamadas
            detecciones = detectar_original(detector, imagen)
            llamadas += 1
            return detecciones if llamadas == 1 else [d for d in detecciones if d.id not in CARTAS]

        args = parser_cli().parse_args(["--simular", "--sin-ventanas", "--frames", "30", "--verificar-simulacion"])
        salida = io.StringIO()
        with patch.object(Detector, "detectar", detectar_sin_cartas_desde_segundo_cuadro), redirect_stdout(salida):
            codigo = ejecutar(args)
        reporte = json.loads(salida.getvalue())
        self.assertEqual(codigo, 2)
        self.assertFalse(reporte["verificacion_ok"])
        self.assertEqual(reporte["cuadros_cartas_incompletas"], 29)

    def test_localiza_marcador_32mm_sin_exigir_cartas(self):
        # Imagen sintética de un único ArUco, entrando por el camino de captura.
        # La cámara ideal y la escala del marcador dan una distancia de 190 mm.
        imagen = np.full((720, 1280, 3), 255, np.uint8)
        marcador = self.detector.generar(0, 160)
        imagen[280:440, 560:720] = cv2.cvtColor(marcador, cv2.COLOR_GRAY2BGR)
        calibracion = Calibracion(self.sim.calibracion.K, np.zeros(5), (1280, 720),
                                  metadatos={"origen": "fixture_prueba"})
        tablero = Tablero("4x4_50", {0: esquinas_cuadrado(32.)}, (32., 32.), es_ejemplo=False)
        fuente = MagicMock()
        blanco = np.full_like(imagen, 255)
        fuente.leer.side_effect = [(blanco, 0, 0.), (imagen, 1, .1),
                                  (blanco, 2, .2), (imagen, 3, .3)]
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal)
            guardar_calibracion(calibracion, ruta / "calibracion.json")
            guardar_tablero(tablero, ruta / "tablero.json")
            args = parser_cli().parse_args([
                "--solo-localizacion", "--cuadrado-ar", "--sin-ventanas", "--frames", "4", "--guardar-video",
                "--calibracion", str(ruta / "calibracion.json"),
                "--tablero", str(ruta / "tablero.json"), "--salida", str(ruta / "salida")])
            with patch("main.FuenteCamara", return_value=fuente), redirect_stdout(io.StringIO()):
                self.assertEqual(ejecutar(args), 0)
            reporte = json.loads((ruta / "salida/reporte.json").read_text())
            lector = cv2.VideoCapture(str(ruta / "salida/camara_cruda.avi"))
            ids_video = []
            try:
                while True:
                    ok, cuadro = lector.read()
                    if not ok:
                        break
                    self.assertEqual(cuadro.shape, imagen.shape)
                    ids_video.append([d.id for d in self.detector.detectar(cuadro)])
            finally:
                lector.release()
        self.assertEqual(ids_video, [[], [0], [], [0]])
        self.assertEqual(reporte["resumen"]["validos"], 2)
        np.testing.assert_array_equal(
            reporte["resumen"]["cuadrado_ar"]["vertices_mundo_mm"], geometria_cuadrado())
        self.assertEqual(reporte["resumen"]["recuperaciones"], 1)
        self.assertEqual(reporte["resumen"]["cuadros_sin_localizacion_inicial"], 1)
        self.assertAlmostEqual(reporte["resumen"]["porcentaje_validos_desde_primera_deteccion"], 200 / 3)
        self.assertIsNone(reporte["cuadros"][2]["T_wc"])
        pose = reporte["cuadros"][1]
        self.assertEqual(pose["ids_usados"], [0])
        self.assertEqual(pose["detecciones"][0]["id"], 0)
        self.assertEqual(np.asarray(pose["detecciones"][0]["esquinas_px"]).shape, (4, 2))
        self.assertEqual(pose["pnp_diagnostico"]["frame_id"], 1)
        self.assertIsNotNone(pose["pnp_diagnostico"]["seleccion"])
        self.assertIsNone(reporte["cuadros"][2]["pnp_diagnostico"]["seleccion"])
        self.assertEqual(pose["cartas"], [])
        self.assertAlmostEqual(pose["T_wc"][2][3], 190., delta=5.)
        fuente.cerrar.assert_called_once()

    def test_vista_externa_se_actualiza_sin_importar_open3d(self):
        args = parser_cli().parse_args([
            "--simular", "--solo-localizacion", "--vista-externa", "--frames", "2"])
        with ExitStack() as context:
            context.enter_context(patch.dict(os.environ, {"DISPLAY": ":prueba"}))
            importador = MagicMock()
            importador.import_module.side_effect = AssertionError("No debe importar Open3D")
            context.enter_context(patch("tp5.visualizador.importlib", importador))
            for funcion in ("namedWindow", "resizeWindow", "moveWindow", "destroyAllWindows"):
                context.enter_context(patch(f"main.cv2.{funcion}"))
            mostrar = context.enter_context(patch("main.cv2.imshow"))
            context.enter_context(patch("main.cv2.waitKey", return_value=-1))
            context.enter_context(patch("main.cv2.getWindowProperty", return_value=1.))
            context.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(ejecutar(args), 0)
        externas = [llamada.args[1] for llamada in mostrar.call_args_list
                    if llamada.args[0] == "TP5 - vista externa (mm)"]
        self.assertEqual(len(externas), 2)
        self.assertEqual(externas[0].shape, (720, 960, 3))
        self.assertGreater(np.count_nonzero(externas[0] != externas[1]), 0)

    @unittest.skipUnless(importlib.util.find_spec("moderngl") and importlib.util.find_spec("trimesh"),
                         "La demo GLB requiere requirements-ar3d.txt")
    def test_demo_glb_cambia_modelo_y_conserva_pose_en_cli(self):
        with tempfile.TemporaryDirectory() as temporal, ExitStack() as context:
            args = parser_cli().parse_args([
                "--simular", "--solo-localizacion", "--demo-yugioh", "--vista-externa",
                "--frames", "3", "--salida", temporal])
            context.enter_context(patch.dict(os.environ, {"DISPLAY": ":prueba"}))
            for funcion in ("namedWindow", "resizeWindow", "moveWindow", "destroyAllWindows"):
                context.enter_context(patch(f"main.cv2.{funcion}"))
            mostrar = context.enter_context(patch("main.cv2.imshow"))
            context.enter_context(patch("main.cv2.waitKey", side_effect=[ord("2"), -1, -1]))
            context.enter_context(patch("main.cv2.getWindowProperty", return_value=1.))
            context.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(ejecutar(args), 0)
            reporte = json.loads((Path(temporal) / "reporte.json").read_text())
        self.assertEqual([r["modelo_3d"] for r in reporte["cuadros"]],
                         ["Dragon Blanco", "Mago Oscuro", "Mago Oscuro"])
        self.assertEqual(len(reporte["resumen"]["modelos_3d"]), 2)
        self.assertEqual(reporte["resumen"]["validos"], 3)
        for r in reporte["cuadros"]:
            verdad = self.sim.generar(r["frame_id"])
            ep, ea = error_transformacion(np.asarray(r["T_wc"]), verdad.T_wc)
            self.assertLess(ep, 5.)
            self.assertLess(ea, 1.)
        camaras = [llamada.args[1] for llamada in mostrar.call_args_list
                   if llamada.args[0] == "TP5 - camara y AR (mm)"]
        self.assertEqual(len(camaras), 3)
        self.assertGreater(np.count_nonzero(camaras[0][150:600] != camaras[1][150:600]), 100)


if __name__ == "__main__":
    unittest.main()
