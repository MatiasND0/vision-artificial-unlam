"""Geometría procedural determinista, sin rig, cámara física ni assets privados."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from tp5.animacion import AnimadorModelo, dibujar_efectos, geometria_efectos
from tp5.modelos import Calibracion, PoseCamara


def modelo_dragon():
    cuerpo = np.float32([[-8, -40, 2], [8, -40, 2], [-8, 40, 20], [8, 40, 20]])
    alas = np.float32([[-3, -5, 20], [-3, -5, 32], [-3, 5, 32],
                       [3, -5, 20], [3, -5, 32], [3, 5, 32]])
    partes = [SimpleNamespace(vertices_mm=v, normales=np.tile(np.float32([0, 0, 1]), (len(v), 1)))
              for v in (cuerpo, alas)]
    return SimpleNamespace(partes=partes, nombre="Blue Eyes White Dragon",
                           ruta=Path("blue_eyes_white_dragon.glb"), tamano_mm=90.)


class PruebasAnimacion(unittest.TestCase):
    def test_alas_se_flexionan_con_cuerpo_y_raices_anclados(self):
        modelo = modelo_dragon()
        animador = AnimadorModelo(modelo)
        inicial = animador.deformar(0.)
        np.testing.assert_array_equal(inicial[0][0], modelo.partes[0].vertices_mm)
        np.testing.assert_allclose(inicial[1][0][[0, 3]], modelo.partes[1].vertices_mm[[0, 3]], atol=1e-6)
        self.assertEqual(np.count_nonzero(animador.mascaras_alas[0]), 0)
        self.assertEqual(np.count_nonzero(animador.mascaras_alas[1]), 4)
        # Las puntas se abren en sentidos opuestos, no arrastran la cabeza/brazos.
        self.assertLess(inicial[1][0][1, 0], modelo.partes[1].vertices_mm[1, 0])
        self.assertGreater(inicial[1][0][4, 0], modelo.partes[1].vertices_mm[4, 0])
        abierto, cerrado = animador.deformar(.6), animador.deformar(1.8)
        self.assertGreater(np.linalg.norm(abierto[1][0][:, :2]-cerrado[1][0][:, :2]), 3)
        np.testing.assert_array_equal(abierto[0][0][:, :2], cerrado[0][0][:, :2])
        np.testing.assert_allclose(abierto[0][0][:, 2]-modelo.partes[0].vertices_mm[:, 2], .3, atol=1e-6)

    def test_periodicidad_normales_y_assets_inmutables(self):
        modelo = modelo_dragon()
        originales = [(p.vertices_mm.copy(), p.normales.copy()) for p in modelo.partes]
        animador = AnimadorModelo(modelo)
        primero, repetido = animador.deformar(.3), animador.deformar(7.5)
        for (v, n), (vr, nr), (vo, no), parte in zip(primero, repetido, originales, modelo.partes):
            np.testing.assert_allclose(v, vr, atol=1e-5)
            np.testing.assert_allclose(n, nr, atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-6)
            np.testing.assert_array_equal(parte.vertices_mm, vo)
            np.testing.assert_array_equal(parte.normales, no)
            self.assertFalse(np.shares_memory(v, parte.vertices_mm))

    def test_mago_respira_sin_mascara_de_alas(self):
        modelo = modelo_dragon()
        modelo.nombre, modelo.ruta = "Mago Oscuro", Path("dark_magician_yugioh.glb")
        animador = AnimadorModelo(modelo)
        self.assertTrue(all(not np.any(m) for m in animador.mascaras_alas))
        for (v, _), parte in zip(animador.deformar(.9), modelo.partes):
            np.testing.assert_array_equal(v[:, :2], parte.vertices_mm[:, :2])
            esperado = 2 + (parte.vertices_mm[:, 2]-2)*1.012 + .6
            np.testing.assert_allclose(v[:, 2], esperado, atol=1e-5)

    def test_efectos_mundo_son_metricos_y_deterministas(self):
        modelo = modelo_dragon()
        a, marcas, particulas, brillo = geometria_efectos(modelo, .5)
        b, _, repetidas, _ = geometria_efectos(modelo, .5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(np.linalg.norm(a[0][:, :2], axis=1), 31.5)
        np.testing.assert_array_equal(a[0][:, 2], .8)
        np.testing.assert_array_equal(particulas, repetidas)
        self.assertEqual(len(marcas), 8)
        self.assertTrue(np.all((brillo >= 0) & (brillo <= 1)))
        self.assertTrue(np.all((particulas[:, 2] >= 1) & (particulas[:, 2] < 39)))
        self.assertGreater(np.linalg.norm(particulas-geometria_efectos(modelo, 1.)[2]), 1)

    def test_efectos_solo_pose_actual_copia_y_tiempo_variable(self):
        cal = Calibracion(np.array([[220., 0, 160], [0, 220., 120], [0, 0, 1]]), np.zeros(5), (320, 240))
        imagen = np.full((240, 320, 3), 25, np.uint8)
        modelo = modelo_dragon()
        T = np.diag([1., -1., -1., 1.]); T[2, 3] = 300
        pose = PoseCamara(0, 0., True, T, np.linalg.inv(T))
        a = dibujar_efectos(imagen, cal, pose, modelo, 0.)
        b = dibujar_efectos(imagen, cal, pose, modelo, .5)
        self.assertGreater(np.count_nonzero(a != imagen), 100)
        self.assertGreater(np.count_nonzero(a != b), 100)
        np.testing.assert_array_equal(imagen, 25)
        perdida = PoseCamara(1, .1, False, ultima_T_wc=pose.T_wc)
        np.testing.assert_array_equal(dibujar_efectos(imagen, cal, perdida, modelo, .5), imagen)
        T_detras = np.eye(4); T_detras[2, 3] = -300
        detras = PoseCamara(2, .2, True, T_detras, np.linalg.inv(T_detras))
        np.testing.assert_array_equal(dibujar_efectos(imagen, cal, detras, modelo, .5), imagen)
        self.assertFalse(np.shares_memory(a, imagen))


if __name__ == "__main__":
    unittest.main()
