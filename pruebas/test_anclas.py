"""Cada marcador conserva su ancla sin convertir una carta móvil en referencia."""
import unittest

import cv2
import numpy as np

from main import procesar_cuadro
from tp5.anclas import marco_marcador, poses_para_render
from tp5.configuracion import CARTAS
from tp5.deteccion import Detector
from tp5.pose import EstimadorPose
from tp5.simulador import Simulador
from tp5.tablero import esquinas_cuadrado


class PruebasAnclas(unittest.TestCase):
    def setUp(self):
        self.sim = Simulador()
        self.detector = Detector()
        self.estimador = EstimadorPose(self.sim.tablero, self.sim.calibracion)

    def procesar(self, cuadro):
        return procesar_cuadro(cuadro.imagen, self.sim.calibracion, self.detector,
                               self.estimador, cuadro.frame_id, cuadro.tiempo_s,
                               {10: CARTAS[10]}, modelos_por_marcador={0: 0, 10: 1})[0]

    def test_marco_fijo_respeta_centro_y_orientacion(self):
        R = cv2.Rodrigues(np.array([.2, -.3, .5]))[0]
        esquinas = esquinas_cuadrado(32) @ R.T + [20, 40, 5]
        T = marco_marcador(esquinas)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], [20, 40, 5])

    def test_mover_mago_no_mueve_camara_ni_dragon(self):
        a = self.procesar(self.sim.generar(0, mover_camara=False))
        b = self.procesar(self.sim.generar(80, mover_camara=False))
        self.assertEqual([i.marcador_id for i in a.instancias_3d], [0, 10])
        self.assertTrue(a.camara.valida and b.camara.valida)
        self.assertNotIn(10, b.camara.ids_usados)
        np.testing.assert_allclose(a.camara.T_wc, b.camara.T_wc, atol=.01)
        np.testing.assert_allclose(a.instancias_3d[0].T_cm, b.instancias_3d[0].T_cm, atol=.01)
        self.assertGreater(np.linalg.norm(a.instancias_3d[1].T_cm[:3, 3]
                                         - b.instancias_3d[1].T_cm[:3, 3]), 5.)

    def test_mago_visible_sin_localizacion_mundial(self):
        self.procesar(self.sim.generar(0))
        perdido = self.procesar(self.sim.generar(1, ocultar_tablero=True))
        self.assertFalse(perdido.camara.valida)
        self.assertIsNone(perdido.camara.T_wc)
        self.assertEqual([i.marcador_id for i in perdido.instancias_3d], [10])
        self.assertIsNone(perdido.instancias_3d[0].T_wm)
        indice, pose_local = poses_para_render(perdido.instancias_3d, perdido.camara.tiempo_s)[0]
        self.assertEqual(indice, 1)
        np.testing.assert_allclose(pose_local.T_cw, perdido.cartas[0].T_cm)
        self.assertIsNone(perdido.camara.T_cw)

    def test_ocultar_mago_no_oculta_dragon(self):
        cuadro = self.sim.generar(0)
        esq = next(d.esquinas for d in cuadro.detecciones if d.id == 10)
        centro = esq.mean(axis=0)
        cv2.fillConvexPoly(cuadro.imagen, np.rint((esq-centro)*1.3+centro).astype(np.int32),
                           (110, 120, 100))
        estado = self.procesar(cuadro)
        self.assertTrue(estado.camara.valida)
        self.assertEqual([i.marcador_id for i in estado.instancias_3d], [0])


if __name__ == "__main__":
    unittest.main()
