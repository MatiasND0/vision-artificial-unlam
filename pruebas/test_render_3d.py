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

    def render(self, modelos, calibracion=None):
        render = RenderizadorModelo(calibracion or self.cal, modelos)
        self.addCleanup(render.cerrar)
        return render

    def pixel(self, puntos, cal=None):
        cal = cal or self.cal
        rvec = cv2.Rodrigues(self.pose.T_cw[:3, :3])[0]
        uv = cv2.projectPoints(np.asarray(puntos, dtype=float).reshape(-1, 3), rvec,
                               self.pose.T_cw[:3, 3], cal.K, cal.dist)[0]
        return np.rint(uv.reshape(-1, 2)).astype(int)

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


if __name__ == "__main__":
    unittest.main()
