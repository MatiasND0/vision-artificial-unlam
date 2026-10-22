"""Combate visual con reloj de cuadros, anclas actuales y daño único por golpe."""
from types import SimpleNamespace
import unittest

import numpy as np

from tp5.duelo import ControlDuelo
from tp5.modelos import InstanciaModelo


def modelos():
    v = np.float32([[-20, -20, 2], [20, -20, 2], [-20, 20, 50], [20, 20, 50]])
    p = SimpleNamespace(vertices_mm=v, triangulos=np.uint32([[0, 1, 2], [1, 3, 2]]))
    return [SimpleNamespace(partes=[p], tamano_mm=90.) for _ in range(2)]


def anclas(frame_id, ids=(0, 10)):
    salida = []
    for id, indice, x in ((0, 0, -80), (10, 1, 80)):
        if id in ids:
            T = np.diag([1., -1., -1., 1.])
            T[:3, 3] = [x, 0, 500]
            salida.append(InstanciaModelo(id, indice, frame_id, T, .1))
    return salida


class PruebasDuelo(unittest.TestCase):
    def setUp(self):
        self.duelo = ControlDuelo(modelos())

    def avanzar(self, hasta, paso=.05, desde=0.):
        resultado = None
        for t in np.arange(desde, hasta+paso/2, paso):
            frame = int(round(t/paso))
            resultado = self.duelo.actualizar(anclas(frame), frame, float(t))
        return resultado

    def test_miran_al_rival_sin_modificar_anclas(self):
        originales = anclas(0)
        copias = [i.T_cm.copy() for i in originales]
        r = self.duelo.actualizar(originales, 0, 0.)
        for n, (indice, pose) in enumerate(r.poses):
            frente_c = pose.T_cw[:3, :3] @ [0., -1., 0.]
            hacia_rival = originales[1-n].T_cm[:3, 3]-originales[n].T_cm[:3, 3]
            hacia_rival /= np.linalg.norm(hacia_rival)
            self.assertGreater(np.dot(frente_c, hacia_rival), .99)
            np.testing.assert_array_equal(originales[n].T_cm, copias[n])
            np.testing.assert_allclose(pose.T_cw @ pose.T_wc, np.eye(4), atol=1e-10)

    def test_no_hay_ataques_sin_los_dos_marcadores(self):
        for n in range(100):
            r = self.duelo.actualizar(anclas(n, (10,)), n, n*.1)
            self.assertEqual([i for i, _ in r.poses], [1])
            self.assertTrue(r.estado["pausado"])
            self.assertIsNone(r.estado["ataque"])
        self.assertEqual(self.duelo.vidas, {"dragon": 100, "mago": 100})
        self.assertEqual(self.duelo.eventos, [])

    def test_vida_cambia_una_sola_vez_al_llegar_el_disparo(self):
        antes = self.avanzar(2.3)
        self.assertEqual(antes.estado["fase"], "dragon_disparo")
        self.assertEqual(antes.estado["vida_mago"], 100)
        golpe = self.avanzar(2.7, desde=2.35)
        self.assertEqual(golpe.estado["fase"], "dragon_impacto")
        self.assertEqual(golpe.estado["vida_mago"], 75)
        self.assertEqual(len(self.duelo.eventos), 1)

    def test_perdida_pausa_reloj_y_no_golpea_un_objetivo_ausente(self):
        antes = self.avanzar(2.)
        for n, t in enumerate(np.arange(2.05, 5.01, .05), start=100):
            r = self.duelo.actualizar(anclas(n, (0,)), n, float(t))
            self.assertEqual(r.estado["tiempo_activo_s"], antes.estado["tiempo_activo_s"])
            self.assertEqual(r.estado["progreso"], antes.estado["progreso"])
            self.assertIsNone(r.estado["ataque"])
            self.assertEqual(r.estado["vida_mago"], 100)
        regreso = self.duelo.actualizar(anclas(200), 200, 5.05)
        self.assertTrue(regreso.estado["pausado"])
        self.assertEqual(regreso.estado["tiempo_activo_s"], antes.estado["tiempo_activo_s"])

    def test_pausa_manual_y_reinicio(self):
        antes = self.avanzar(1.2)
        self.duelo.alternar_pausa()
        r = self.duelo.actualizar(anclas(30), 30, 1.3)
        self.assertTrue(r.estado["pausado"])
        self.assertEqual(r.estado["tiempo_activo_s"], antes.estado["tiempo_activo_s"])
        self.duelo.reiniciar()
        self.assertFalse(self.duelo.pausa_manual)
        self.assertEqual(self.duelo.vidas, {"dragon": 100, "mago": 100})
        self.assertEqual(self.duelo.eventos, [])

    def test_salto_de_tiempo_no_adelanta_danos(self):
        antes = self.avanzar(1.)
        despues = self.duelo.actualizar(anclas(200), 200, 30.)
        self.assertTrue(despues.estado["pausado"])
        self.assertEqual(despues.estado["tiempo_activo_s"], antes.estado["tiempo_activo_s"])
        self.assertEqual(self.duelo.eventos, [])

    def test_resultado_independiente_de_frecuencia_de_cuadros(self):
        a = self.avanzar(8., paso=.05)
        self.duelo = ControlDuelo(modelos())
        b = self.avanzar(8., paso=.1)
        self.assertEqual(a.estado["vida_dragon"], b.estado["vida_dragon"])
        self.assertEqual(a.estado["vida_mago"], b.estado["vida_mago"])
        self.assertEqual(a.estado["fase"], b.estado["fase"])
        self.assertAlmostEqual(a.estado["progreso"], b.estado["progreso"], places=10)

    def test_victoria_conserva_vidas_y_no_repite_impactos(self):
        r = self.avanzar(25.)
        self.assertEqual(r.estado["fase"], "victoria")
        self.assertEqual(r.estado["ganador"], "dragon")
        self.assertEqual(r.estado["vida_mago"], 0)
        self.assertEqual(r.estado["vida_dragon"], 40)
        self.assertEqual(len(self.duelo.eventos), 7)
        self.assertIsNone(r.estado["ataque"])

    def test_ignora_anclas_de_otro_cuadro_y_ids_duplicados(self):
        r = self.duelo.actualizar(anclas(0), 1, .1)
        self.assertEqual(r.poses, [])
        repetidas = anclas(2) + anclas(2, (0,))
        r = self.duelo.actualizar(repetidas, 2, .2)
        self.assertEqual([i for i, _ in r.poses], [1])
        self.assertTrue(r.estado["pausado"])


if __name__ == "__main__":
    unittest.main()
