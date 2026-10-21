"""Regresiones de imagen para el rescate de umbral, sin datos personales."""
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from tp5.deteccion import Detector


class PruebasDeteccion(unittest.TestCase):
    def test_rescata_patron_degradado_con_geometria_conocida(self):
        # Caso reproducible: patrón pequeño, desenfoque, bajo contraste y ruido.
        # Solo contiene un marcador generado; no deriva de una captura real.
        imagen = np.full((240, 240), 255, np.uint8)
        base = Detector(rescate_umbral=False)
        imagen[104:136, 104:136] = base.generar(0, 32)
        rotacion = cv2.getRotationMatrix2D((120, 120), 15, 1)
        imagen = cv2.warpAffine(imagen, rotacion, (240, 240),
                               flags=cv2.INTER_LINEAR, borderValue=255)
        imagen = cv2.GaussianBlur(imagen, (0, 0), 1.2)
        ruido = np.random.default_rng(1234).normal(0, 10, imagen.shape)
        imagen = np.uint8(np.clip(100 + .2 * imagen + ruido, 0, 255))

        self.assertEqual(base.detectar(imagen), [])
        self.assertEqual(Detector().detectar(imagen), [])
        detectadas = Detector(rescate_umbral=True).detectar(imagen)
        self.assertEqual([d.id for d in detectadas], [0])
        originales = np.float32([[[104, 104], [135, 104], [135, 135], [104, 135]]])
        verdad = cv2.transform(originales, rotacion)[0]
        errores = np.linalg.norm(detectadas[0].esquinas - verdad, axis=1)
        # Comprobar el marcador y su orden de esquinas, no solo un ID devuelto.
        self.assertLess(errores.max(), 2.)

    def test_conserva_detecciones_y_refinamiento_sin_ejecutar_rescate(self):
        imagen = np.full((240, 320, 3), 255, np.uint8)
        generador = Detector(rescate_umbral=False)
        for identificador, x, y in [(0, 35, 40), (11, 185, 115)]:
            patron = generador.generar(identificador, 64, borde_px=16)
            imagen[y:y + 96, x:x + 96] = patron[..., None]
        for refinar in (False, True):
            with self.subTest(refinar_bordes=refinar):
                base = Detector(refinar_bordes=refinar, rescate_umbral=False)
                por_defecto = Detector(refinar_bordes=refinar)
                self.assertIsNone(por_defecto._detector_rescate)
                con_rescate = Detector(refinar_bordes=refinar, rescate_umbral=True)
                segunda_pasada = Mock()
                segunda_pasada.detectMarkers.side_effect = AssertionError(
                    "El rescate no debe ejecutarse si la base detectó marcadores")
                con_rescate._detector_rescate = segunda_pasada
                esperadas = base.detectar(imagen)
                originales = por_defecto.detectar(imagen)
                obtenidas = con_rescate.detectar(imagen)
                self.assertEqual([d.id for d in esperadas], [0, 11])
                self.assertEqual([d.id for d in originales], [0, 11])
                self.assertEqual([d.id for d in obtenidas], [0, 11])
                for esperada, original, obtenida in zip(esperadas, originales, obtenidas):
                    np.testing.assert_array_equal(original.esquinas, esperada.esquinas)
                    np.testing.assert_array_equal(obtenida.esquinas, esperada.esquinas)
                segunda_pasada.detectMarkers.assert_not_called()

    def test_negativos_no_generan_identificadores(self):
        negativos = [np.full((240, 320), valor, np.uint8)
                     for valor in (0, 32, 127, 224, 255)]
        y, x = np.indices((240, 320))
        negativos.extend(np.uint8(((x // lado + y // lado) % 2) * 255)
                         for lado in (8, 15, 32, 80))
        rng = np.random.default_rng(1234)
        negativos.extend(rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
                         for _ in range(8))
        for rescatar in (False, True):
            detector = Detector(rescate_umbral=rescatar)
            for indice, imagen in enumerate(negativos):
                with self.subTest(rescate_umbral=rescatar, negativo=indice):
                    self.assertEqual(detector.detectar(imagen), [])


if __name__ == "__main__":
    unittest.main()
