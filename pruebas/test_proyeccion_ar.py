"""Pruebas geométricas sintéticas: no requieren detector, Open3D ni pantalla."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tp5.ar import dibujar_ar, geometria_carta, geometria_cuadrado, proyectar_puntos
from tp5.modelos import Calibracion, CartaConfig, EstadoMundo, PoseCamara, PoseCarta, Tablero
from tp5.visualizador import Visualizador3D, puntos_frustum


def transformacion(rvec=(0., 0., 0.), t=(0., 0., 700.)):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.array(rvec, dtype=float))
    T[:3, 3] = t
    return T


class PruebasGeometricas(unittest.TestCase):
    def setUp(self):
        self.cal = Calibracion(np.array([[800., 0, 640], [0, 810, 360], [0, 0, 1]]),
                               np.array([.10, -.04, .002, -.001, .005]), (1280, 720),
                               .2, {"sintetica": True})
        self.tablero = Tablero("4x4_50", {0: np.array([[-30, 30, 0], [30, 30, 0],
                                                      [30, -30, 0], [-30, -30, 0]])},
                               (600., 400.))
        self.config = {10: CartaConfig(10, "Mago Oscuro", 40., 2500, 2100)}
        self.imagen = np.full((720, 1280, 3), 80, np.uint8)
        self.T_cm = transformacion((2.3, .1, .1), (0, 60, 700))

    def estado(self, frame=0, valida=True, cartas=True, t=(0., 0., 700.)):
        T_wc = transformacion((np.pi, 0, 0), t)
        if valida:
            p = PoseCamara(frame, frame*.1, True, np.linalg.inv(T_wc), T_wc,
                           (0,), .1)
        else:
            p = PoseCamara(frame, frame*.1, False, motivo="sin tablero",
                           ultima_T_wc=T_wc, antiguedad_s=.1)
        lista = [PoseCarta(10, frame, self.T_cm.copy(), .2)] if cartas else []
        return EstadoMundo(p, lista)

    def test_proyeccion_igual_a_opencv_con_distorsion(self):
        puntos = np.array([[-120, 90, 0], [100, -90, 10], [10, 50, 70], [0, 0, 0]])
        pixeles, visibles = proyectar_puntos(puntos, self.T_cm, self.cal)
        rvec, _ = cv2.Rodrigues(self.T_cm[:3, :3])
        esperados, _ = cv2.projectPoints(puntos.astype(float), rvec, self.T_cm[:3, 3],
                                         self.cal.K, self.cal.dist)
        self.assertTrue(visibles.all())
        np.testing.assert_allclose(pixeles, esperados.reshape(-1, 2), atol=1e-10)
        sin_dist, _ = cv2.projectPoints(puntos.astype(float), rvec, self.T_cm[:3, 3],
                                        self.cal.K, np.zeros(5))
        self.assertGreater(np.max(np.abs(pixeles-sin_dist.reshape(-1, 2))), .1)

    def test_profundidad_y_transformacion_invalidas(self):
        pixeles, visibles = proyectar_puntos([[0, 0, -10], [0, 0, 0], [0, 0, 10]],
                                            np.eye(4), self.cal)
        np.testing.assert_array_equal(visibles, [False, False, True])
        self.assertTrue(np.isnan(pixeles[:2]).all())
        T = np.eye(4); T[0, 0] = 2
        with self.assertRaises(ValueError):
            proyectar_puntos([[0, 0, 1]], T, self.cal)

    def test_geometria_base_canonica_y_altura_positiva(self):
        geometria = geometria_carta(40)
        np.testing.assert_array_equal(geometria["base"],
                                       [[-20, 20, 0], [20, 20, 0],
                                        [20, -20, 0], [-20, -20, 0]])
        self.assertEqual(np.max(geometria["vertical"][:, 2]), 72)
        self.assertEqual(np.min(geometria["vertical"][:, 2]), 0)

    def test_geometria_cuadrado_mundial_y_parametros(self):
        np.testing.assert_array_equal(geometria_cuadrado(),
                                      [[-20, 20, 20], [20, 20, 20],
                                       [20, -20, 20], [-20, -20, 20]])
        personalizado = geometria_cuadrado(12, 0)
        np.testing.assert_array_equal(personalizado[:, 2], 0)
        np.testing.assert_allclose(np.linalg.norm(personalizado -
                                   np.roll(personalizado, 1, axis=0), axis=1), 12)
        for lado in (0, -1, np.nan, np.inf):
            with self.subTest(lado=lado), self.assertRaises(ValueError):
                geometria_cuadrado(lado)
        for altura in (-1, np.nan, np.inf):
            with self.subTest(altura=altura), self.assertRaises(ValueError):
                geometria_cuadrado(altura_mm=altura)

    def test_cuadrado_mundial_se_reproyecta_con_camara_movil(self):
        original = self.imagen.copy()
        puntos = np.array([[-20, 20, 20], [20, 20, 20],
                           [20, -20, 20], [-20, -20, 20]], dtype=float)
        proyecciones = []
        for frame, posicion in enumerate(((0, 0, 400), (30, -20, 320))):
            T_wc = transformacion((np.pi + .06*frame, .04*frame, .03*frame), posicion)
            T_cw = np.linalg.inv(T_wc)
            camara = PoseCamara(frame, frame*.1, True, T_cw, T_wc, (0,), .1)
            estado = EstadoMundo(camara)
            rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
            esperados = cv2.projectPoints(puntos, rvec, T_cw[:3, 3],
                                          self.cal.K, self.cal.dist)[0].reshape(4, 2)
            proyectados, visibles = proyectar_puntos(geometria_cuadrado(), T_cw, self.cal)
            self.assertTrue(visibles.all())
            np.testing.assert_allclose(proyectados, esperados, atol=1e-10)
            proyecciones.append(proyectados)

            salida = dibujar_ar(self.imagen, self.cal, estado, {}, cuadrado_ar=True)
            sin_cuadrado = dibujar_ar(self.imagen, self.cal, estado, {})
            cambio = np.any(salida != sin_cuadrado, axis=2)
            cambio[:160] = False  # excluir las líneas distintas del panel
            y, x = np.nonzero(cambio)
            self.assertGreater(len(x), 100)
            # Los límites dibujados coinciden con la proyección independiente.
            np.testing.assert_allclose([x.min(), x.max(), y.min(), y.max()],
                                       [esperados[:, 0].min(), esperados[:, 0].max(),
                                        esperados[:, 1].min(), esperados[:, 1].max()], atol=3)
            centro = np.rint(esperados.mean(axis=0)).astype(int)
            self.assertGreater(salida[centro[1], centro[0], 1], original[centro[1], centro[0], 1])
            np.testing.assert_array_equal(self.imagen, original)
            self.assertFalse(np.shares_memory(salida, self.imagen))
        self.assertGreater(np.linalg.norm(proyecciones[1] - proyecciones[0]), 20)

    def test_cuadrado_desaparece_sin_pose_actual_aunque_haya_ultima(self):
        estado = self.estado(valida=False, cartas=False)
        self.assertIsNotNone(estado.camara.ultima_T_wc)
        with patch("tp5.ar._panel") as panel:
            salida = dibujar_ar(self.imagen, self.cal, estado, {}, cuadrado_ar=True)
        np.testing.assert_array_equal(salida, self.imagen)
        lineas = panel.call_args.args[1]
        self.assertIn("SEGUIMIENTO PERDIDO", lineas[0])
        self.assertIn("40x40 mm", lineas[-1])
        self.assertIn("+20 mm", lineas[-1])
        self.assertIn("oculto: sin pose mundial actual", lineas[-1])

    def test_cuadrado_no_rellena_vertices_detras_o_cruzando_camara(self):
        poses = [transformacion(t=(0, 0, -700)),
                 transformacion((0, np.pi/2, 0), (0, 0, 10))]
        for T_cw in poses:
            with self.subTest(T_cw=T_cw):
                camara = PoseCamara(0, 0., True, T_cw, np.linalg.inv(T_cw), (0,), .1)
                estado = EstadoMundo(camara)
                with patch("tp5.ar._panel"):
                    salida = dibujar_ar(self.imagen, self.cal, estado, {}, cuadrado_ar=True)
                np.testing.assert_array_equal(salida, self.imagen)

    def test_ar_copia_imagen_y_valida_resolucion(self):
        original = self.imagen.copy()
        salida = dibujar_ar(self.imagen, self.cal, self.estado(), self.config)
        np.testing.assert_array_equal(original, self.imagen)
        self.assertFalse(np.shares_memory(salida, self.imagen))
        self.assertGreater(np.count_nonzero(salida[120:] != original[120:]), 100)
        with self.assertRaises(ValueError):
            dibujar_ar(self.imagen[:360, :640], self.cal, self.estado(), self.config)

    def test_ar_descarta_cartas_de_otro_cuadro(self):
        estado = self.estado(frame=3)
        estado.cartas[0].frame_id = 2
        salida = dibujar_ar(self.imagen, self.cal, estado, self.config)
        sin_cartas = dibujar_ar(self.imagen, self.cal, self.estado(3, cartas=False), self.config)
        np.testing.assert_array_equal(salida, sin_cartas)

    def test_ar_no_dibuja_carta_detras_de_camara(self):
        estado = self.estado()
        estado.cartas[0].T_cm = transformacion(t=(0, 0, -700))
        salida = dibujar_ar(self.imagen, self.cal, estado, self.config)
        sin_cartas = dibujar_ar(self.imagen, self.cal, self.estado(cartas=False), self.config)
        np.testing.assert_array_equal(salida, sin_cartas)

    def test_ar_actual_funciona_relativa_a_camara_sin_tablero(self):
        estado = self.estado(valida=False)
        salida = dibujar_ar(self.imagen, self.cal, estado, self.config)
        sin_cartas = dibujar_ar(self.imagen, self.cal, self.estado(valida=False, cartas=False), self.config)
        self.assertGreater(np.count_nonzero(salida[120:] != sin_cartas[120:]), 100)

    def test_frustum_respeta_T_wc_y_Z_optico(self):
        T_wc = transformacion((np.pi, 0, 0), (50, -100, 700))
        frustum = puntos_frustum(self.cal, T_wc, 120)
        np.testing.assert_allclose(frustum[0], [50, -100, 700], atol=1e-10)
        # +Z de cámara apunta al tablero, hacia -Z mundial.
        np.testing.assert_allclose(frustum[1:, 2], 580, atol=1e-10)
        puntos_c = (frustum-T_wc[:3, 3]) @ T_wc[:3, :3]
        reprojectados, _ = cv2.projectPoints(puntos_c[1:], np.zeros(3), np.zeros(3),
                                            self.cal.K, self.cal.dist)
        np.testing.assert_allclose(reprojectados.reshape(-1, 2),
                                   [[0, 0], [1279, 0], [1279, 719], [0, 719]], atol=.01)

    def test_modo_deshabilitado_no_importa_open3d(self):
        with patch("tp5.visualizador.importlib.import_module", side_effect=AssertionError("importó Open3D")):
            v = Visualizador3D(self.tablero, self.cal, habilitado=False)
            self.assertTrue(v.actualizar(self.estado()))
            self.assertEqual(v.renderizar_vista_externa().shape, (720, 960, 3))
            v.cerrar()
            self.assertFalse(v.actualizar(self.estado(1)))

    def test_error_claro_si_falta_open3d(self):
        with patch("tp5.visualizador.importlib.import_module", side_effect=ImportError("sin módulo")):
            with self.assertRaisesRegex(RuntimeError, "Open3D no está disponible"):
                Visualizador3D(self.tablero, self.cal)

    def test_perdida_recuperacion_no_extiende_ni_conecta_trayectoria(self):
        v = Visualizador3D(self.tablero, self.cal, habilitado=False)
        v.actualizar(self.estado(0))
        v.actualizar(self.estado(1, t=(10, 0, 700)))
        v.actualizar(self.estado(1, t=(10, 0, 700)))  # refrescar ventana no duplica
        self.assertEqual(len(v.datos_escena["trayectoria"]), 2)
        v.actualizar(self.estado(2, valida=False))
        self.assertFalse(v.datos_escena["camara_valida"])
        self.assertIsNone(v.datos_escena["T_wc_actual"])
        self.assertIsNotNone(v.datos_escena["T_wc_mostrada"])
        self.assertEqual(len(v.datos_escena["trayectoria"]), 2)
        self.assertEqual(v.datos_escena["cartas"], {})
        v.actualizar(self.estado(3, t=(50, 0, 700)))
        self.assertTrue(v.datos_escena["camara_valida"])
        np.testing.assert_array_equal(v.datos_escena["tramos_trayectoria"], [[0, 1]])
        v.actualizar(self.estado(4, t=(60, 0, 700)))
        np.testing.assert_array_equal(v.datos_escena["tramos_trayectoria"], [[0, 1], [2, 3]])

    def test_direccion_camara_composicion_carta_y_cuadro(self):
        v = Visualizador3D(self.tablero, self.cal, habilitado=False)
        estado = self.estado()
        v.actualizar(estado)
        direccion = v.datos_escena["direccion"]
        np.testing.assert_allclose(direccion[1]-direccion[0], [0, 0, -180], atol=1e-10)
        np.testing.assert_allclose(v.datos_escena["cartas"][10], estado.camara.T_wc @ self.T_cm)
        estado.cartas[0].frame_id = -1
        v.actualizar(estado)
        self.assertEqual(v.datos_escena["cartas"], {})

    def test_trayectoria_acotada(self):
        v = Visualizador3D(self.tablero, self.cal, habilitado=False)
        for frame in range(605):
            v.actualizar(self.estado(frame, cartas=False, t=(frame, 0, 700)))
        self.assertEqual(len(v.datos_escena["trayectoria"]), 600)
        self.assertEqual(v.datos_escena["trayectoria"][0, 0], 5)


if __name__ == "__main__":
    unittest.main()
