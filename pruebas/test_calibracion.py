"""Verificación sintética; no constituye una calibración física de webcam."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from tp5.calibracion import (
    calibrar_desde_imagenes, calibrar_desde_puntos, cargar_calibracion,
    detectar_patron, guardar_calibracion, guardar_patron_svg, puntos_patron,
)
from tp5.modelos import Calibracion


def vistas_sinteticas(ruido_px=0.0):
    """Cámara conocida con distorsión no nula, vistas inclinadas y cámara móvil."""
    rng = np.random.default_rng(2125)
    K = np.array([[1000., 0., 640.], [0., 970., 360.], [0., 0., 1.]])
    dist = np.array([-0.12, 0.035, 0.001, -0.0007, 0.006])
    objeto = puntos_patron((9, 6), 25.).astype(np.float64)
    objeto -= objeto.mean(axis=0)
    objetos, imagenes = [], []
    while len(objetos) < 26:
        rotacion = rng.uniform([-.6, -.6, -.3], [.6, .6, .3])
        traslacion = rng.uniform([-235., -140., 490.], [235., 140., 920.])
        proyeccion, _ = cv2.projectPoints(objeto, rotacion, traslacion, K, dist)
        imagen = proyeccion.reshape(-1, 2)
        if np.any(imagen < [8., 8.]) or np.any(imagen >= [1272., 712.]):
            continue
        objetos.append(objeto.copy())
        imagenes.append(imagen + rng.normal(0., ruido_px, imagen.shape))
    return objetos, imagenes, K, dist


class CalibracionSinteticaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objetos, cls.imagenes, cls.K, cls.dist = vistas_sinteticas()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.calibracion = calibrar_desde_puntos(cls.objetos, cls.imagenes, (1280, 720),
                                                  {"origen": "sintetico", "lado_casillero_mm": 25.})

    def test_recupera_intrinsecos_y_distorsion_conocidos(self):
        np.testing.assert_allclose(self.calibracion.K, self.K, atol=.02, rtol=0)
        np.testing.assert_allclose(self.calibracion.dist, self.dist, atol=.002, rtol=0)
        self.assertLess(self.calibracion.rms_px, 1e-3)
        self.assertEqual(self.calibracion.metadatos["vistas"], 26)
        self.assertEqual(self.calibracion.metadatos["imagen"], "cruda_sin_undistort")

    def test_calibracion_con_ruido_subpixel(self):
        objetos, imagenes, K, dist = vistas_sinteticas(ruido_px=.08)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            calibracion = calibrar_desde_puntos(objetos, imagenes, (1280, 720))
        self.assertLess(calibracion.rms_px, .15)
        np.testing.assert_allclose(calibracion.K, K, atol=3., rtol=0)
        self.assertLess(abs(calibracion.dist[0] - dist[0]), .02)

    def test_roundtrip_preserva_parametros_unidades_y_metadatos(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "sesion" / "calibracion.json"
            guardar_calibracion(self.calibracion, ruta)
            datos = json.loads(ruta.read_text())
            self.assertEqual(datos["version"], 1)
            self.assertEqual(datos["unidades"], "mm")
            cargada = cargar_calibracion(ruta)
            np.testing.assert_array_equal(cargada.K, self.calibracion.K)
            np.testing.assert_array_equal(cargada.dist, self.calibracion.dist)
            self.assertEqual(cargada.tamano, (1280, 720))
            self.assertEqual(cargada.metadatos, self.calibracion.metadatos)
            cargada.validar_imagen(np.zeros((720, 1280, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "Resolución"):
                cargada.validar_imagen(np.zeros((480, 640, 3), np.uint8))

    def test_rechaza_formato_incompatible_e_intrinsecos_invalidos(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "calibracion.json"
            guardar_calibracion(self.calibracion, ruta)
            original = json.loads(ruta.read_text())
            casos = [("version", 2), ("unidades", "m"), ("K", [[0] * 3] * 3),
                     ("dist", [float("nan")] * 5), ("metadatos", []), ("tamano", [0, 720])]
            for clave, valor in casos:
                with self.subTest(clave=clave):
                    datos = {**original, clave: valor}
                    ruta.write_text(json.dumps(datos))
                    with self.assertRaises(ValueError):
                        cargar_calibracion(ruta)
            ruta.write_text('{"version": 1, "unidades": "mm", "metadatos": {}}')
            with self.assertRaisesRegex(ValueError, "incompleto"):
                cargar_calibracion(ruta)

    def test_rechaza_puntos_y_vistas_degeneradas(self):
        with self.assertRaisesRegex(ValueError, "al menos 6"):
            calibrar_desde_puntos(self.objetos[:5], self.imagenes[:5], (1280, 720))
        with self.assertRaisesRegex(ValueError, "vistas distintas"):
            calibrar_desde_puntos([self.objetos[0]] * 10, [self.imagenes[0]] * 10, (1280, 720))
        with self.assertRaisesRegex(ValueError, "resolución declarada"):
            calibrar_desde_puntos(self.objetos, self.imagenes, (320, 240))
        with self.assertRaisesRegex(ValueError, "cantidades distintas"):
            calibrar_desde_puntos(self.objetos, [self.imagenes[0][:-1]] + self.imagenes[1:], (1280, 720))
        corrupta = self.imagenes[0].copy()
        corrupta[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finitos"):
            calibrar_desde_puntos(self.objetos, [corrupta] + self.imagenes[1:], (1280, 720))

    def test_advierte_pocas_vistas(self):
        with self.assertWarnsRegex(UserWarning, "15 y 25"):
            calibracion = calibrar_desde_puntos(self.objetos[:8], self.imagenes[:8], (1280, 720))
        self.assertTrue(calibracion.metadatos["recomendaciones"])


class PatronEImagenesTests(unittest.TestCase):
    def test_patron_en_mm_y_geometria(self):
        puntos = puntos_patron((9, 6), 25.)
        self.assertEqual(puntos.shape, (54, 3))
        np.testing.assert_array_equal(puntos[1] - puntos[0], [25, 0, 0])
        np.testing.assert_array_equal(puntos[9] - puntos[0], [0, 25, 0])
        for patron, lado in [((0, 6), 25), ((9, 2), 25), ((9, 6), 0), ((9, 6), -1), ((9, 6), float("nan"))]:
            with self.subTest(patron=patron, lado=lado), self.assertRaises(ValueError):
                puntos_patron(patron, lado)
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "patron.svg"
            guardar_patron_svg(ruta, (9, 6), 25.)
            raiz = ET.parse(ruta).getroot()
            self.assertEqual(raiz.attrib["width"], "270mm")
            self.assertEqual(raiz.attrib["height"], "205mm")
            self.assertIn("100%", ruta.read_text())

    def test_detecta_tablero_renderizado(self):
        imagen = np.full((480, 680, 3), 255, np.uint8)
        for fila in range(7):
            for columna in range(10):
                if (fila + columna) % 2 == 0:
                    x, y = 90 + columna * 50, 65 + fila * 50
                    imagen[y:y + 50, x:x + 50] = 0
        esquinas = detectar_patron(imagen, (9, 6))
        self.assertIsNotNone(esquinas)
        self.assertEqual(esquinas.shape, (54, 2))
        esperado = np.array([(90 + x * 50 - .5, 65 + y * 50 - .5)
                             for y in range(1, 7) for x in range(1, 10)])
        # OpenCV puede escoger cualquiera de las dos orientaciones equivalentes.
        error = min(np.max(np.abs(esquinas - esperado)), np.max(np.abs(esquinas[::-1] - esperado)))
        self.assertLess(error, .2)

    def test_rechaza_resoluciones_mezcladas_aun_sin_patron(self):
        with tempfile.TemporaryDirectory() as temporal:
            rutas = [Path(temporal) / "a.png", Path(temporal) / "b.png"]
            cv2.imwrite(str(rutas[0]), np.zeros((480, 640, 3), np.uint8))
            cv2.imwrite(str(rutas[1]), np.zeros((720, 1280, 3), np.uint8))
            with patch("tp5.calibracion.detectar_patron", return_value=None):
                with self.assertRaisesRegex(ValueError, "Resolución no uniforme"):
                    calibrar_desde_imagenes(rutas)

    def test_rechaza_metadatos_no_finitos_al_guardar(self):
        calibracion = Calibracion(np.diag([500., 500., 1.]), np.zeros(5), (640, 480),
                                 metadatos={"temperatura": float("nan")})
        with tempfile.TemporaryDirectory() as temporal, self.assertRaisesRegex(ValueError, "JSON finitos"):
            guardar_calibracion(calibracion, Path(temporal) / "invalida.json")


if __name__ == "__main__":
    unittest.main()
