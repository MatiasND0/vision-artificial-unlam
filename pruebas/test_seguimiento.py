"""El rescate sigue evidencia de la imagen actual y expira sin una lectura nueva."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tp5.deteccion import Detector
from tp5.seguimiento import DetectorContinuo


class PruebasSeguimiento(unittest.TestCase):
    def setUp(self):
        self.detector = DetectorContinuo(ids_fijos=[0])
        self.imagen = np.full((360, 480, 3), 235, np.uint8)
        marcador = self.detector.generar(0, 120)
        self.imagen[120:240, 180:300] = cv2.cvtColor(marcador, cv2.COLOR_GRAY2BGR)

    def test_roi_redetecta_actual_sin_cambiar_id_ni_orden(self):
        inicial = self.detector.detectar(self.imagen, 0.)
        self.assertEqual([d.id for d in inicial], [0])
        siguiente = cv2.warpAffine(self.imagen, np.float32([[1, 0, 7], [0, 1, -4]]),
                                    (480, 360), borderValue=(235, 235, 235))
        with patch.object(Detector, "detectar", return_value=[]):
            recuperada = self.detector.detectar(siguiente, .1)
        self.assertEqual([d.id for d in recuperada], [0])
        self.assertEqual(self.detector.fuentes[0], "aruco_roi")
        np.testing.assert_allclose(recuperada[0].esquinas,
                                   inicial[0].esquinas + [7, -4], atol=1.)

    def test_flujo_mide_movimiento_del_cuadro_actual(self):
        inicial = self.detector.detectar(self.imagen, 0.)
        siguiente = cv2.warpAffine(self.imagen, np.float32([[1, 0, 4], [0, 1, 3]]),
                                    (480, 360), borderValue=(235, 235, 235))
        with patch.object(Detector, "detectar", return_value=[]), \
                patch.object(self.detector, "_roi", return_value=None):
            recuperada = self.detector.detectar(siguiente, .1)
        self.assertEqual([d.id for d in recuperada], [0])
        self.assertEqual(self.detector.fuentes[0], "flujo_verificado")
        np.testing.assert_allclose(recuperada[0].esquinas,
                                   inicial[0].esquinas + [4, 3], atol=.5)

    def test_no_inventa_deteccion_al_ocultar_el_marcador(self):
        self.detector.detectar(self.imagen, 0.)
        blanco = np.full_like(self.imagen, 235)
        self.assertEqual(self.detector.detectar(blanco, .1), [])
        self.assertEqual(self.detector.fuentes, {})

    def test_rescate_movil_no_agrega_su_id_a_las_referencias_fijas(self):
        detector = DetectorContinuo(ids_fijos=[0], ids_moviles=[10])
        imagen = self.imagen.copy()
        imagen[120:240, 180:300] = cv2.cvtColor(detector.generar(10, 120), cv2.COLOR_GRAY2BGR)
        detector.detectar(imagen, 0.)
        with patch.object(Detector, "detectar", return_value=[]):
            rescatada = detector.detectar(imagen, .1)
        self.assertEqual([d.id for d in rescatada], [10])
        self.assertEqual(detector.ids_fijos, {0})
        self.assertEqual(detector.fuentes[10], "aruco_roi")

    def test_flujo_expira_sin_redecodificar(self):
        self.detector.detectar(self.imagen, 0.)
        with patch.object(Detector, "detectar", return_value=[]), \
                patch.object(self.detector, "_roi", return_value=None):
            self.assertEqual(len(self.detector.detectar(self.imagen, .1)), 1)
            self.assertEqual(len(self.detector.detectar(self.imagen, .2)), 1)
            self.assertEqual(self.detector.detectar(self.imagen, .31), [])

    def test_no_propaga_id_si_la_imagen_muestra_otro(self):
        self.detector.detectar(self.imagen, 0.)
        otro = self.imagen.copy()
        otro[120:240, 180:300] = cv2.cvtColor(self.detector.generar(1, 120), cv2.COLOR_GRAY2BGR)
        with patch.object(Detector, "detectar", return_value=[]), \
                patch.object(self.detector, "_roi", return_value=None):
            self.assertEqual(self.detector.detectar(otro, .1), [])


if __name__ == "__main__":
    unittest.main()
