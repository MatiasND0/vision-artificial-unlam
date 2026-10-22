"""Pruebas reales de EGL con mallas generadas, sin assets ni cámara física."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from tp5.modelos import Calibracion, PoseCamara
from tp5.render_3d import RenderizadorModelo, matriz_proyeccion


def malla(color=(1., 1., 1., 1.), z=0., lado=80., textura=None,
          alpha_modo="OPAQUE", alpha_corte=.5, centro=(0., 0.)):
    l = lado / 2
    vertices = np.array([[-l, l, z], [l, l, z], [l, -l, z], [-l, -l, z]], np.float32)
    vertices[:, :2] += centro
    if textura is None:
        textura = np.full((2, 2, 4), 255, np.uint8)
    return SimpleNamespace(vertices_mm=vertices,
                           normales=np.tile(np.float32([0, 0, 1]), (4, 1)),
                           uv=np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
                           triangulos=np.uint32([[0, 1, 2], [0, 2, 3]]),
                           textura_rgba=textura, color_rgba=np.float32(color),
                           alpha_modo=alpha_modo, alpha_corte=alpha_corte)


def modelo(partes, nombre="sintetico"):
    return SimpleNamespace(ruta=Path(nombre + ".glb"), nombre=nombre, partes=partes,
                           tamano_mm=80., atribucion={"origen": "geometria sintetica"})


class PruebasRender3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import moderngl
            contexto = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception as exc:
            raise unittest.SkipTest(f"No se pudo crear contexto EGL real: {exc}")
        cls.gpu = contexto.info["GL_RENDERER"]
        contexto.release()

    def setUp(self):
        self.cal = Calibracion(np.array([[220., 0, 147.], [0, 225., 103.], [0, 0, 1]]),
                               np.zeros(5), (320, 240))
        self.imagen = np.full((240, 320, 3), (40, 80, 120), np.uint8)
        T_cw = np.diag([1., -1., -1., 1.])
        T_cw[:3, 3] = [0, 0, 300]
        self.pose = PoseCamara(0, 0., True, T_cw, np.linalg.inv(T_cw), (0,), .1)

    def render(self, modelos, calibracion=None, **opciones):
        render = RenderizadorModelo(calibracion or self.cal, modelos, **opciones)
        self.addCleanup(render.cerrar)
        return render

    def pixel(self, puntos, cal=None):
        cal = cal or self.cal
        rvec = cv2.Rodrigues(self.pose.T_cw[:3, :3])[0]
        uv = cv2.projectPoints(np.asarray(puntos, dtype=float).reshape(-1, 3), rvec,
                               self.pose.T_cw[:3, 3], cal.K, cal.dist)[0]
        return np.rint(uv.reshape(-1, 2)).astype(int)

    def ancla(self, x=0., profundidad=300., tiempo=0.):
        T = self.pose.T_cw.copy()
        T[:3, 3] = [x, 0, profundidad]
        return PoseCamara(0, tiempo, True, T, np.linalg.inv(T), (0,), .1)

    def test_matriz_gpu_coincide_con_proyeccion_opencv(self):
        # Transform feedback ejecuta de verdad la matriz en GPU, sin rasterizar.
        render = self.render([modelo([malla()])])
        ctx = render._ctx
        render._fbo.use()
        programa = ctx.program(vertex_shader="""
            #version 330
            uniform mat4 P;
            in vec3 punto;
            out vec4 clip;
            void main() { clip = P * vec4(punto, 1.0); gl_Position = clip; }
            """, varyings=["clip"])
        self.addCleanup(programa.release)
        puntos = np.float32([[0, 0, 100], [-30, 20, 200], [50, -30, 700]])
        entrada = ctx.buffer(puntos.tobytes())
        self.addCleanup(entrada.release)
        salida = ctx.buffer(reserve=len(puntos)*4*4)
        self.addCleanup(salida.release)
        vao = ctx.vertex_array(programa, [(entrada, "3f", "punto")])
        self.addCleanup(vao.release)
        programa["P"].write(matriz_proyeccion(self.cal).T.tobytes())
        vao.transform(salida, mode=render._gl.POINTS, vertices=len(puntos))
        clip = np.frombuffer(salida.read(), np.float32).reshape(-1, 4)
        ndc = clip[:, :2] / clip[:, 3:4]
        pixeles = np.column_stack([(ndc[:, 0]+1)*160-.5,
                                   (1-ndc[:, 1])*120-.5])
        esperados = cv2.projectPoints(puntos, np.zeros(3), np.zeros(3),
                                      self.cal.K, self.cal.dist)[0].reshape(-1, 2)
        np.testing.assert_allclose(pixeles, esperados, atol=3e-5)

    def test_zbuffer_oculta_malla_lejana_en_ambos_ordenes(self):
        cerca = malla((1, 0, 0, 1), z=30.)
        lejos = malla((0, 0, 1, 1), z=0.)
        resultados = []
        for partes in ([cerca, lejos], [lejos, cerca]):
            render = self.render([modelo(partes)])
            imagen = render.dibujar(self.imagen, self.pose)
            x, y = self.pixel([[0, 0, 30]])[0]
            self.assertGreater(imagen[y, x, 2], 200)
            self.assertLess(imagen[y, x, 0], 5)
            resultados.append(imagen)
            render.cerrar()
        np.testing.assert_array_equal(resultados[0], resultados[1])

    def test_uv_y_framebuffer_conservan_orientacion_y_canales(self):
        textura = np.full((16, 16, 4), 255, np.uint8)
        textura[:8, :8, :3] = [255, 0, 0]  # arriba izquierda: rojo
        textura[:8, 8:, :3] = [0, 255, 0]
        textura[8:, :8, :3] = [0, 0, 255]
        textura[8:, 8:, :3] = [255, 255, 0]
        render = self.render([modelo([malla(textura=textura)])])
        salida = render.dibujar(self.imagen, self.pose)
        pixeles = self.pixel([[-20, 20, 0], [20, 20, 0], [-20, -20, 0], [20, -20, 0]])
        colores = np.array([salida[y, x] for x, y in pixeles])
        self.assertEqual(list(np.argmax(colores[:3], axis=1)), [2, 1, 0])
        self.assertGreater(colores[3, 1], 200)
        self.assertGreater(colores[3, 2], 200)
        self.assertLess(colores[3, 0], 5)

    def test_mask_descarta_alpha_antes_de_escribir_profundidad(self):
        textura = np.full((16, 16, 4), 255, np.uint8)
        textura[..., :3] = [255, 0, 0]
        textura[:, 8:, 3] = 100  # menor que corte 0,5; no tapa la malla azul
        cerca = malla(z=30., textura=textura, alpha_modo="MASK")
        fondo = malla((0, 0, 1, 1))
        render = self.render([modelo([cerca, fondo])])
        salida = render.dibujar(self.imagen, self.pose)
        izquierda, derecha = self.pixel([[-20, 0, 30], [20, 0, 30]])
        color_izq = salida[izquierda[1], izquierda[0]]
        color_der = salida[derecha[1], derecha[0]]
        self.assertGreater(color_izq[2], 200)
        self.assertGreater(color_der[0], 200)
        self.assertLess(color_der[2], 5)

    def test_blend_premultiplicado_y_fondo_transparente(self):
        textura = np.full((2, 2, 4), [255, 0, 0, 128], np.uint8)
        render = self.render([modelo([malla(textura=textura, alpha_modo="BLEND")])])
        original = self.imagen.copy()
        salida = render.dibujar(self.imagen, self.pose)
        x, y = self.pixel([[0, 0, 0]])[0]
        self.assertTrue(170 < salida[y, x, 2] < 185)
        self.assertTrue(15 < salida[y, x, 0] < 25)
        self.assertTrue(35 < salida[y, x, 1] < 45)
        np.testing.assert_array_equal(salida[0, 0], original[0, 0])
        np.testing.assert_array_equal(self.imagen, original)
        self.assertFalse(np.shares_memory(salida, self.imagen))

    def test_overlay_distorsionado_coincide_opencv_sin_remapear_camara(self):
        self.cal.dist[:] = [.4, -.08, .001, -.002, .01]
        pequeno = malla((1, 0, 0, 1), lado=8., centro=(120., 60.))
        render = self.render([modelo([pequeno])])
        salida = render.dibujar(self.imagen, self.pose)
        cambio = np.any(salida != self.imagen, axis=2)
        y, x = np.nonzero(cambio)
        self.assertGreater(len(x), 9)
        rvec = cv2.Rodrigues(self.pose.T_cw[:3, :3])[0]
        centro = np.array([[120., 60., 0.]])
        esperado = cv2.projectPoints(centro, rvec, self.pose.T_cw[:3, 3],
                                     self.cal.K, self.cal.dist)[0].reshape(2)
        ideal = cv2.projectPoints(centro, rvec, self.pose.T_cw[:3, 3],
                                  self.cal.K, np.zeros(5))[0].reshape(2)
        self.assertGreater(np.linalg.norm(esperado-ideal), 5)
        np.testing.assert_allclose([x.mean(), y.mean()], esperado, atol=1.2)
        np.testing.assert_array_equal(salida[:30], self.imagen[:30])

    def test_pose_perdida_no_reutiliza_overlay_y_descarta_detras(self):
        render = self.render([modelo([malla((1, 0, 0, 1))])])
        visible = render.dibujar(self.imagen, self.pose)
        self.assertGreater(np.count_nonzero(visible != self.imagen), 100)
        perdida = PoseCamara(1, .1, False, ultima_T_wc=self.pose.T_wc, antiguedad_s=.1)
        salida = render.dibujar(self.imagen, perdida)
        np.testing.assert_array_equal(salida, self.imagen)
        self.assertFalse(np.shares_memory(salida, self.imagen))
        T = np.eye(4)
        T[2, 3] = -300
        detras = PoseCamara(2, .2, True, T, np.linalg.inv(T), (0,), .1)
        np.testing.assert_array_equal(render.dibujar(self.imagen, detras), self.imagen)

    def test_cambio_modelo_validacion_y_cierre_idempotente(self):
        rojo = modelo([malla((1, 0, 0, 1))], "rojo")
        verde = modelo([malla((0, 1, 0, 1))], "verde")
        render = self.render([rojo, verde])
        self.assertIs(render.modelo_actual, rojo)
        x, y = self.pixel([[0, 0, 0]])[0]
        self.assertGreater(render.dibujar(self.imagen, self.pose)[y, x, 2], 200)
        render.seleccionar(1)
        self.assertIs(render.modelo_actual, verde)
        self.assertGreater(render.dibujar(self.imagen, self.pose)[y, x, 1], 200)
        with self.assertRaises(IndexError):
            render.seleccionar(2)
        with self.assertRaises(ValueError):
            render.dibujar(self.imagen[:120], self.pose)
        render.cerrar()
        render.cerrar()
        with self.assertRaises(RuntimeError):
            render.dibujar(self.imagen, self.pose)

    def test_animacion_gpu_varia_alas_y_es_reproducible_con_tiempo_relativo(self):
        cuerpo = malla((.4, .4, .4, 1))
        cuerpo.vertices_mm[:, 1] *= 1.5
        alas = malla((0, 1, 0, 1))
        alas.vertices_mm = np.float32([[-5, 0, 18], [-5, 0, 40], [-10, 8, 40],
                                      [5, 0, 18], [5, 0, 40], [10, 8, 40]])
        alas.normales = np.tile(np.float32([0, 0, 1]), (6, 1))
        alas.uv = np.zeros((6, 2), np.float32)
        alas.triangulos = np.uint32([[0, 1, 2], [3, 4, 5]])
        dragon = modelo([cuerpo, alas], "blue_eyes_white_dragon")
        originales = [p.vertices_mm.copy() for p in dragon.partes]
        render = self.render([dragon])
        self.pose.tiempo_s = 100.
        estatico = render.dibujar(self.imagen, self.pose)
        self.pose.tiempo_s = 100.6
        np.testing.assert_array_equal(render.dibujar(self.imagen, self.pose), estatico)
        render.cerrar()
        resultados = []
        for origen in (100., 900.):
            render = self.render([dragon], animar=True)
            self.pose.tiempo_s = origen
            a = render.dibujar(self.imagen, self.pose)
            self.pose.tiempo_s = origen+.6
            b = render.dibujar(self.imagen, self.pose)
            self.assertGreater(np.count_nonzero(a != b), 10)
            resultados.append(b)
            render.cerrar()
        np.testing.assert_array_equal(resultados[0], resultados[1])
        for parte, original in zip(dragon.partes, originales):
            np.testing.assert_array_equal(parte.vertices_mm, original)

    def test_efectos_flag_se_componen_detras_del_modelo_y_desaparecen(self):
        dragon = modelo([malla((1, 0, 0, 1), lado=20, z=10)], "blue_eyes_white_dragon")
        render = self.render([dragon])
        base = render.dibujar(self.imagen, self.pose)
        render.cerrar()
        render = self.render([dragon], efectos=True)
        efecto = render.dibujar(self.imagen, self.pose)
        self.assertGreater(np.count_nonzero(efecto != base), 100)
        x, y = self.pixel([[0, 0, 10]])[0]
        np.testing.assert_array_equal(efecto[y-3:y+4, x-3:x+4], base[y-3:y+4, x-3:x+4])
        perdida = PoseCamara(1, .1, False, ultima_T_wc=self.pose.T_wc)
        np.testing.assert_array_equal(render.dibujar(self.imagen, perdida), self.imagen)

    def test_dos_instancias_identicas_en_anclas_distintas_y_seleccion_intacta(self):
        rojo = modelo([malla((1, 0, 0, 1), lado=30)], "rojo")
        verde = modelo([malla((0, 1, 0, 1), lado=30)], "verde")
        render = self.render([rojo, verde])
        render.seleccionar(1)
        izquierda, derecha = self.ancla(-60), self.ancla(60)
        original = self.imagen.copy()
        salida = render.dibujar_instancias(self.imagen, [(0, izquierda), (1, derecha)])
        for pose, canal in ((izquierda, 2), (derecha, 1)):
            rvec = cv2.Rodrigues(pose.T_cw[:3, :3])[0]
            punto = cv2.projectPoints(np.zeros((1, 3)), rvec, pose.T_cw[:3, 3],
                                      self.cal.K, self.cal.dist)[0].reshape(2)
            x, y = np.rint(punto).astype(int)
            self.assertGreater(salida[y, x, canal], 200)
        self.assertIs(render.modelo_actual, verde)
        np.testing.assert_array_equal(self.imagen, original)
        self.assertFalse(np.shares_memory(salida, self.imagen))
        # La API anterior sigue dibujando el modelo seleccionado, sin cambiarlo.
        np.testing.assert_array_equal(render.dibujar(self.imagen, derecha),
                                      render.dibujar_instancias(self.imagen, [(1, derecha)]))

    def test_zbuffer_compartido_entre_instancias_que_se_intersectan(self):
        # Dos planos inclinados se cruzan: uno está delante a la izquierda y
        # el otro a la derecha. Componer renders separados no resuelve esto.
        rojo = modelo([malla((1, 0, 0, 1))], "rojo")
        azul = modelo([malla((0, 0, 1, 1))], "azul")
        render = self.render([rojo, azul])
        poses = []
        for angulo, z in ((.5, 300.), (-.5, 300.7)):
            T = self.pose.T_cw.copy()
            T[:3, :3] = T[:3, :3] @ cv2.Rodrigues(np.array([0., angulo, 0.]))[0]
            T[2, 3] = z
            poses.append(PoseCamara(0, 0., True, T, np.linalg.inv(T)))
        a = render.dibujar_instancias(self.imagen, [(0, poses[0]), (1, poses[1])])
        b = render.dibujar_instancias(self.imagen, [(1, poses[1]), (0, poses[0])])
        np.testing.assert_array_equal(a, b)
        cy, cx = int(self.cal.K[1, 2]), int(self.cal.K[0, 2])
        self.assertGreater(a[cy, cx-12, 2], 200)
        self.assertLess(a[cy, cx-12, 0], 5)
        self.assertGreater(a[cy, cx+12, 0], 200)
        self.assertLess(a[cy, cx+12, 2], 5)

    def test_instancia_perdida_se_oculta_independientemente_y_valida_antes_de_gpu(self):
        render = self.render([modelo([malla(lado=20)], "blue_eyes_white_dragon"),
                              modelo([malla(lado=20)], "dark_magician_yugioh")], efectos=True)
        izquierda, derecha = self.ancla(-60), self.ancla(60)
        perdida = PoseCamara(1, 0., False, ultima_T_wc=derecha.T_wc)
        solo = render.dibujar_instancias(self.imagen, [(0, izquierda)])
        mezclado = render.dibujar_instancias(self.imagen, [(0, izquierda), (1, perdida)])
        np.testing.assert_array_equal(solo, mezclado)
        np.testing.assert_array_equal(render.dibujar_instancias(self.imagen, []), self.imagen)
        np.testing.assert_array_equal(render.dibujar_instancias(self.imagen, [(0, perdida), (1, perdida)]),
                                      self.imagen)
        contenido = render._fbo.read(components=4)
        with self.assertRaises(IndexError):
            render.dibujar_instancias(self.imagen, [(0, izquierda), (2, perdida)])
        T = derecha.T_cw.copy(); T[0, 0] = 2  # no pertenece a SO(3)
        mala = PoseCamara(0, 0., True, T, np.eye(4))
        with self.assertRaises(ValueError):
            render.dibujar_instancias(self.imagen, [(0, izquierda), (1, mala)])
        self.assertEqual(render._fbo.read(components=4), contenido)
        # Fallar la validación no cierra ni deja parcialmente dibujada la escena.
        np.testing.assert_array_equal(render.dibujar_instancias(self.imagen, [(0, izquierda)]), solo)

    def test_animacion_mismo_modelo_en_dos_instancias_no_comparte_fase(self):
        inclinado = malla((1, 0, 0, 1), lado=35)
        inclinado.vertices_mm[:, 2] = [0, 60, 60, 0]
        mago = modelo([inclinado], "dark_magician_yugioh")
        original = inclinado.vertices_mm.copy()
        render = self.render([mago], animar=True)
        render.dibujar_instancias(self.imagen, [(0, self.ancla(-50, 180, 0))])
        izquierda, derecha = self.ancla(-50, 180, .9), self.ancla(50, 180, 2.7)
        juntas = render.dibujar_instancias(self.imagen, [(0, izquierda), (0, derecha)])
        solo_izq = render.dibujar_instancias(self.imagen, [(0, izquierda)])
        solo_der = render.dibujar_instancias(self.imagen, [(0, derecha)])
        np.testing.assert_array_equal(juntas[:, :140], solo_izq[:, :140])
        np.testing.assert_array_equal(juntas[:, 155:], solo_der[:, 155:])
        otra_fase = self.ancla(-50, 180, 2.7)
        self.assertGreater(np.count_nonzero(render.dibujar_instancias(self.imagen, [(0, otra_fase)]) != solo_izq), 10)
        np.testing.assert_array_equal(inclinado.vertices_mm, original)

    def test_efectos_y_animacion_por_ancla_conservan_paletas_y_tiempo(self):
        dragon = modelo([malla((1, 0, 0, 1), lado=16, z=10)], "blue_eyes_white_dragon")
        mago = modelo([malla((0, 1, 0, 1), lado=16, z=10)], "dark_magician_yugioh")
        render = self.render([dragon, mago], animar=True, efectos=True)
        izquierda, derecha = self.ancla(-60), self.ancla(60)
        a = render.dibujar_instancias(self.imagen, [(0, izquierda), (1, derecha)])
        incrementos = []
        for pose in (izquierda, derecha):
            rvec = cv2.Rodrigues(pose.T_cw[:3, :3])[0]
            uv = cv2.projectPoints(np.array([[28., 0., .8]]), rvec, pose.T_cw[:3, 3],
                                   self.cal.K, self.cal.dist)[0].reshape(2)
            x, y = np.rint(uv).astype(int)
            incrementos.append(a[y, x].astype(int)-self.imagen[y, x])
        self.assertGreater(incrementos[0][1], incrementos[0][2]*1.5)  # cian
        self.assertGreater(incrementos[1][2], incrementos[1][1]*1.5)  # violeta
        izquierda.tiempo_s = derecha.tiempo_s = .6
        b = render.dibujar_instancias(self.imagen, [(0, izquierda), (1, derecha)])
        self.assertGreater(np.count_nonzero(a[:, :140] != b[:, :140]), 100)
        self.assertGreater(np.count_nonzero(a[:, 155:] != b[:, 155:]), 100)


if __name__ == "__main__":
    unittest.main()
