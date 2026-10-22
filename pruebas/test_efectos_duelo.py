"""Proyección y presentación del combate con estados y cámaras sintéticos."""
import copy
import unittest

import cv2
import numpy as np

from tp5.efectos_duelo import dibujar_ataques, dibujar_hud
from tp5.modelos import Calibracion


def estado(tipo="disparo", atacante="mago", progreso=.5):
    return {"fase": f"{tipo}_{atacante}", "progreso": progreso, "pausado": False,
            "motivo": "", "vida_dragon": 100, "vida_mago": 100, "ganador": None,
            "tiempo_activo_s": 4.,
            "ataque": {"tipo": tipo, "atacante": atacante, "progreso": progreso,
                        "origen_c": [-60., 0., 300.], "destino_c": [60., 0., 400.]}}


class PruebasEfectosDuelo(unittest.TestCase):
    def setUp(self):
        self.cal = Calibracion(np.array([[240., 0, 160.], [0, 235., 120.], [0, 0, 1.]]),
                               np.array([.2, -.05, .001, -.002, .005]), (320, 240))
        self.imagen = np.full((240, 320, 3), 30, np.uint8)

    def test_orbe_avanza_al_rival_en_3d_y_se_proyecta_con_distorsion(self):
        centros = []
        for progreso in (.2, .8):
            datos = estado(progreso=progreso)
            salida = dibujar_ataques(self.imagen, self.cal, datos)
            # El núcleo blanco distingue al orbe de su cola violeta.
            blancos = np.min(salida, axis=2) > 200
            y, x = np.nonzero(blancos)
            self.assertGreater(len(x), 0)
            centro = np.array([x.mean(), y.mean()])
            centros.append(centro)
            ataque = datos["ataque"]
            origen, destino = np.array(ataque["origen_c"]), np.array(ataque["destino_c"])
            punto = origen + progreso*(destino-origen)
            punto[1] -= 8*np.sin(np.pi*progreso)
            esperado = cv2.projectPoints(punto[None], np.zeros(3), np.zeros(3),
                                         self.cal.K, self.cal.dist)[0].reshape(2)
            np.testing.assert_allclose(centro, esperado, atol=2.5)
        self.assertGreater(centros[1][0]-centros[0][0], 40)

    def test_rayo_cian_y_orbe_violeta_tienen_colores_y_colas_distintos(self):
        spans = []
        incrementos = []
        for atacante in ("dragon", "mago"):
            datos = estado(atacante=atacante, progreso=.65)
            datos["ataque"]["destino_c"][2] = 300.
            salida = dibujar_ataques(self.imagen, self.cal, datos)
            diferencia = salida.astype(int)-self.imagen.astype(int)
            incrementos.append(diferencia.sum(axis=(0, 1)))
            y, x = np.nonzero(diferencia.max(axis=2) > 25)
            spans.append(x.max()-x.min())
            self.assertGreater(len(y), 20)
        self.assertGreater(incrementos[0][1], incrementos[0][2])
        self.assertGreater(incrementos[1][2], incrementos[1][1])
        self.assertGreater(spans[0]-spans[1], 15)

    def test_capas_separan_carga_disparo_e_impacto_y_estallido_se_apaga(self):
        for tipo in ("carga", "disparo", "impacto"):
            datos = estado(tipo=tipo, progreso=.4)
            fondo = dibujar_ataques(self.imagen, self.cal, datos)
            frente = dibujar_ataques(self.imagen, self.cal, datos, frente=True)
            visible, invisible = (frente, fondo) if tipo == "impacto" else (fondo, frente)
            self.assertGreater(np.count_nonzero(visible != self.imagen), 50)
            np.testing.assert_array_equal(invisible, self.imagen)
        final = estado(tipo="impacto", progreso=1.)
        np.testing.assert_array_equal(dibujar_ataques(self.imagen, self.cal, final, frente=True),
                                      self.imagen)

    def test_pausa_y_ausencia_de_ataque_no_dejan_overlay(self):
        pausado = estado(); pausado["pausado"] = True
        sin_ataque = estado(); sin_ataque["ataque"] = None
        for datos in (pausado, sin_ataque):
            for frente in (False, True):
                salida = dibujar_ataques(self.imagen, self.cal, datos, frente=frente)
                np.testing.assert_array_equal(salida, self.imagen)
                self.assertFalse(np.shares_memory(salida, self.imagen))

    def test_descarta_profundidad_no_finita_y_coordenadas_extremas(self):
        for tipo in ("carga", "disparo", "impacto"):
            for valor in (-10., 0., np.nan, np.inf):
                datos = estado(tipo=tipo)
                datos["ataque"]["origen_c"][2] = valor
                with self.subTest(tipo=tipo, profundidad=valor):
                    salida = dibujar_ataques(self.imagen, self.cal, datos, frente=(tipo == "impacto"))
                    np.testing.assert_array_equal(salida, self.imagen)
            extremos = estado(tipo=tipo)
            extremos["ataque"]["origen_c"] = [1e12, 0, 1]
            extremos["ataque"]["destino_c"] = [1e12+100, 0, 1]
            np.testing.assert_array_equal(dibujar_ataques(self.imagen, self.cal, extremos,
                                                        frente=(tipo == "impacto")), self.imagen)

    def test_efectos_son_deterministas_y_no_mutan_imagen_ni_estado(self):
        datos = estado(tipo="impacto", atacante="dragon", progreso=.3)
        anterior = copy.deepcopy(datos)
        original = self.imagen.copy()
        a = dibujar_ataques(self.imagen, self.cal, datos, frente=True)
        dibujar_ataques(self.imagen, self.cal, estado(progreso=.8))
        b = dibujar_ataques(self.imagen, self.cal, datos, frente=True)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.imagen, original)
        self.assertEqual(datos, anterior)
        self.assertFalse(np.shares_memory(a, self.imagen))

    def test_hud_barras_independientes_y_fraccion_de_vida(self):
        datos = estado(); datos["ataque"] = None
        anterior = copy.deepcopy(datos)
        lleno = dibujar_hud(self.imagen, datos)
        datos["vida_dragon"] = 20
        bajo = dibujar_hud(self.imagen, datos)
        self.assertGreater(lleno[130, 80, 1], 150)
        self.assertLess(bajo[130, 80, 1], 100)
        np.testing.assert_array_equal(lleno[126:135, 170:300], bajo[126:135, 170:300])
        np.testing.assert_array_equal(lleno[:105], self.imagen[:105])
        np.testing.assert_array_equal(lleno[175:], self.imagen[175:])
        np.testing.assert_array_equal(self.imagen, 30)
        self.assertEqual(anterior["vida_dragon"], 100)
        self.assertEqual(datos["vida_dragon"], 20)
        self.assertFalse(np.shares_memory(lleno, self.imagen))

    def test_hud_pausa_motivo_y_victoria_legibles_en_imagen_pequena(self):
        datos = estado(); datos["ataque"] = None
        normal = dibujar_hud(self.imagen, datos)
        datos["pausado"] = True; datos["motivo"] = "Mostrá ambos marcadores"
        pausado = dibujar_hud(self.imagen, datos)
        self.assertGreater(np.count_nonzero(normal[140:174] != pausado[140:174]), 20)
        datos["ganador"] = "dragon"; datos["vida_mago"] = 0
        ganador = dibujar_hud(self.imagen, datos)
        self.assertEqual(ganador.shape, (240, 320, 3))
        self.assertGreater(np.count_nonzero(ganador[180:] != self.imagen[180:]), 100)
        np.testing.assert_array_equal(ganador[180:, :40], self.imagen[180:, :40])
        np.testing.assert_array_equal(ganador, dibujar_hud(self.imagen, datos))


if __name__ == "__main__":
    unittest.main()
