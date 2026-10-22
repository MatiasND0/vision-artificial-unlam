"""Mesa AR TP5: cámara cruda → ArUco → PnP → mundo 3D y cartas AR."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from tp5.ar import dibujar_ar, geometria_cuadrado
from tp5.anclas import crear_instancias, poses_para_render
from tp5.calibracion import cargar_calibracion
from tp5.configuracion import CARTAS
from tp5.deteccion import Detector
from tp5.fuentes import FuenteCamara
from tp5.modelos import CartaConfig, EstadoMundo
from tp5.pose import EstimadorPose, estimar_cartas
from tp5.simulador import Simulador
from tp5.seguimiento import DetectorContinuo
from tp5.tablero import cargar_tablero, crear_tablero_ejemplo
from tp5.visualizador import Visualizador3D


# Tolerancias del ensayo raster, fijadas antes de medir; no aplican a cámara real.
LIMITES_SINTETICOS = {"posicion_mm": 5.0, "orientacion_grados": 1.0,
                      "carta_mm": 12.0}
FPS_VIDEO = 10.0


def procesar_cuadro(imagen, calibracion, detector, estimador, frame_id, tiempo_s,
                    cartas_config=CARTAS, *, cuadrado_ar=False, renderizador_modelo=None,
                    modelos_por_marcador=None):
    calibracion.validar_imagen(imagen)
    detecciones = (detector.detectar(imagen, tiempo_s) if isinstance(detector, DetectorContinuo)
                   else detector.detectar(imagen))
    pose = estimador.estimar(detecciones, frame_id, tiempo_s)
    cartas = estimar_cartas(detecciones, calibracion, cartas_config, pose)
    estado = EstadoMundo(pose, cartas, detecciones)
    fondo = imagen
    if modelos_por_marcador is not None:
        estado.instancias_3d = crear_instancias(estado, estimador.tablero, modelos_por_marcador)
        if renderizador_modelo is not None:
            fondo = renderizador_modelo.dibujar_instancias(
                imagen, poses_para_render(estado.instancias_3d, tiempo_s))
    elif renderizador_modelo is not None:
        fondo = renderizador_modelo.dibujar(imagen, pose)
    return estado, dibujar_ar(fondo, calibracion, estado,
                             {} if modelos_por_marcador is not None else cartas_config,
                             cuadrado_ar=cuadrado_ar)


def error_transformacion(estimada, verdadera):
    posicion = float(np.linalg.norm(estimada[:3, 3] - verdadera[:3, 3]))
    coseno = (np.trace(estimada[:3, :3] @ verdadera[:3, :3].T) - 1) / 2
    angulo = float(np.degrees(np.arccos(np.clip(coseno, -1, 1))))
    return posicion, angulo


def cargar_cartas(ruta):
    if ruta is None:
        return CARTAS
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if datos.get("unidades") != "mm" or datos.get("version") != 1:
        raise ValueError("Cartas: se requiere JSON versión 1 y unidades mm")
    cartas = {int(c["id"]): CartaConfig(**c) for c in datos["cartas"]}
    if not cartas or any(c.lado_mm <= 0 or not np.isfinite(c.lado_mm) for c in cartas.values()):
        raise ValueError("Se requieren tamaños de cartas positivos en mm")
    return cartas


def parser_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simular", action="store_true", help="usar cámara sintética y geometría de ejemplo")
    p.add_argument("--sin-ventanas", action="store_true", help="ejecución sin GUI ni Open3D")
    vistas = p.add_mutually_exclusive_group()
    vistas.add_argument("--sin-3d", action="store_true", help="solo ventana de cámara/AR")
    vistas.add_argument("--vista-externa", action="store_true",
                        help="mostrar la escena 3D proyectada en otra ventana, sin Open3D")
    p.add_argument("--solo-localizacion", action="store_true",
                   help="localizar la cámara con referencias fijas, sin configurar cartas")
    p.add_argument("--cuadrado-ar", action="store_true",
                   help="proyectar un cuadrado de 40 mm a 20 mm del origen del tablero, hacia +Z")
    modelos = p.add_mutually_exclusive_group()
    modelos.add_argument("--modelo-3d", type=Path, action="append",
                         help="GLB estático sobre el origen; repetir para alternar con teclas 1–9")
    modelos.add_argument("--demo-yugioh", action="store_true",
                         help="usar Dragón Blanco y Mago Oscuro de assets/modelos; teclas 1 y 2")
    p.add_argument("--por-marcador", action="store_true",
                   help="mostrar Dragón sobre ID 0 y Mago sobre ID 10, simultáneamente; incluye demo Yu-Gi-Oh")
    p.add_argument("--lado-mago-mm", type=float,
                   help="lado negro medido del ArUco ID 10, en mm; predeterminado 32 (40 al simular)")
    p.add_argument("--seguimiento-basico", action="store_true",
                   help="desactivar búsqueda localizada, flujo verificado y control de saltos")
    p.add_argument("--sin-animacion", action="store_true", help="mostrar los modelos en pose estática")
    p.add_argument("--sin-efectos", action="store_true", help="ocultar círculo de invocación y partículas")
    p.add_argument("--tamano-modelo-mm", type=float, default=90.,
                   help="mayor dimensión del modelo en mm (predeterminado: 90)")
    p.add_argument("--giro-modelo-grados", type=float, default=0.,
                   help="giro del modelo alrededor del eje Z del marcador")
    p.add_argument("--frames", type=int, default=0, help="límite de cuadros; 0: 180 en simulación, ilimitados en webcam")
    p.add_argument("--salida", type=Path, help="guardar PNG y reporte JSON del ensayo")
    p.add_argument("--guardar-video", action="store_true",
                   help="guardar también video crudo MJPEG para revisión; requiere --salida")
    p.add_argument("--probar-perdida", action="store_true", help="ocultar tablero durante un tramo de simulación")
    p.add_argument("--verificar-simulacion", action="store_true", help="salir con error si falla el ensayo raster")
    p.add_argument("--calibracion", type=Path, help="JSON de calibración real")
    p.add_argument("--tablero", type=Path, help="JSON con esquinas medidas en mundo, mm")
    p.add_argument("--cartas", type=Path, help="JSON con IDs, lados medidos y estadísticas")
    p.add_argument("--camara", type=int, default=0)
    return p


def ejecutar(args):
    if args.por_marcador:
        if args.modelo_3d or args.cartas:
            raise ValueError("--por-marcador usa los dos modelos incluidos y el ID 10; no combinar con --modelo-3d/--cartas")
        args.demo_yugioh = True
        args.solo_localizacion = True
    if args.frames < 0:
        raise ValueError("--frames debe ser >= 0")
    if args.guardar_video and not args.salida:
        raise ValueError("--guardar-video requiere --salida para conservar la grabación")
    if (args.probar_perdida or args.verificar_simulacion) and not args.simular:
        raise ValueError("Las opciones de ensayo sintético requieren --simular")
    if args.verificar_simulacion and args.frames and args.frames < 30:
        raise ValueError("La verificación de movimiento requiere al menos 30 cuadros")
    if not args.sin_ventanas and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No hay sesión gráfica; usar --sin-ventanas para el ensayo sin GUI")
    fuente = None
    visor = None
    renderizador_modelo = None
    video = None
    ruta_video = None
    if args.solo_localizacion and args.cartas:
        raise ValueError("--solo-localizacion no usa --cartas; quitar una de esas opciones")
    cartas_config = {} if args.solo_localizacion else cargar_cartas(args.cartas)
    tablero = cargar_tablero(args.tablero) if args.tablero else crear_tablero_ejemplo()
    modelos_por_marcador = None
    if args.por_marcador:
        if 0 not in tablero.marcadores:
            raise ValueError("--por-marcador requiere un tablero con referencia fija ID 0")
        lado_mago = args.lado_mago_mm if args.lado_mago_mm is not None else (40. if args.simular else 32.)
        if not np.isfinite(lado_mago) or lado_mago <= 0:
            raise ValueError("El lado del marcador del Mago debe ser positivo, en mm")
        cartas_config = {10: CartaConfig(10, "Mago Oscuro", lado_mago, 2500, 2100, (220, 80, 200))}
        modelos_por_marcador = {0: 0, 10: 1}
    if set(tablero.marcadores) & set(cartas_config):
        raise ValueError("Los IDs de cartas y tablero deben ser distintos")
    sim = None
    if args.simular:
        if args.calibracion or args.cartas:
            raise ValueError("El simulador usa sus intrínsecos y cartas conocidos; no acepta calibración/cartas reales")
        sim = Simulador(tablero=tablero)
        calibracion = sim.calibracion
    else:
        if not (args.calibracion and args.tablero):
            raise ValueError("Webcam requiere --calibracion y --tablero con medidas reales; ver README.md")
        if not args.solo_localizacion and not args.cartas:
            raise ValueError("Webcam con AR requiere --cartas medidas; para probar solo cámara usar --solo-localizacion")
        calibracion = cargar_calibracion(args.calibracion)
        if calibracion.metadatos.get("sintetica") or calibracion.metadatos.get("origen") == "sintetica":
            raise ValueError("La calibración sintética no sirve para una webcam real")
        if tablero.es_ejemplo:
            raise ValueError("Medir el tablero y marcar es_ejemplo=false antes de usar webcam")
        if args.cartas and json.loads(args.cartas.read_text(encoding="utf-8")).get("es_ejemplo", True):
            raise ValueError("Medir los marcadores de cartas y marcar es_ejemplo=false antes de usar webcam")
    seguimiento_continuo = not args.seguimiento_basico and not args.simular
    detector = (DetectorContinuo(tablero.diccionario, ids_fijos=tablero.marcadores,
                                 ids_moviles=cartas_config if args.por_marcador else ())
                if seguimiento_continuo else Detector(tablero.diccionario))
    estimador = EstimadorPose(tablero, calibracion,
                              verificar_continuidad=seguimiento_continuo)
    limite = args.frames or (180 if args.simular else 0)
    registros = []
    posiciones, angulos, errores_cartas = [], [], []
    perdidas_esperadas = 0
    falsas_validas = 0
    recuperadas = 0
    cuadros_cartas_incompletas = 0
    antes_perdida = False
    hubo_localizacion = False
    inicio = time.monotonic()
    try:
        rutas_modelos = args.modelo_3d or []
        if args.demo_yugioh:
            carpeta = Path(__file__).resolve().parent / "assets/modelos"
            rutas_modelos = [carpeta / "blue_eyes_white_dragon.glb",
                            carpeta / "dark_magician_yugioh.glb"]
        modelos_cargados = []
        if rutas_modelos:
            if len(rutas_modelos) > 9:
                raise ValueError("Se pueden alternar hasta 9 modelos con el teclado")
            from tp5.activos_3d import cargar_modelo
            from tp5.render_3d import RenderizadorModelo
            modelos_cargados = [cargar_modelo(ruta, args.tamano_modelo_mm,
                                              args.giro_modelo_grados)
                                for ruta in rutas_modelos]
            if args.demo_yugioh:
                modelos_cargados[0].nombre = "Dragon Blanco"
                modelos_cargados[1].nombre = "Mago Oscuro"
            renderizador_modelo = RenderizadorModelo(
                calibracion, modelos_cargados, animar=args.demo_yugioh and not args.sin_animacion,
                efectos=args.demo_yugioh and not args.sin_efectos)
        visor = Visualizador3D(tablero, calibracion,
                              habilitado=not(args.sin_ventanas or args.sin_3d or args.vista_externa))
        if args.guardar_video:
            args.salida.mkdir(parents=True, exist_ok=True)
            ruta_video = args.salida / "camara_cruda.avi"
            if ruta_video.exists():
                raise ValueError("Ya existe un video en --salida; usar otra carpeta para conservar ambas sesiones")
            video = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*"MJPG"),
                                    FPS_VIDEO, calibracion.tamano)
            if not video.isOpened():
                raise RuntimeError("No se pudo crear el video MJPEG")
        if not sim:
            fuente = FuenteCamara(args.camara, calibracion.tamano)
        if args.vista_externa and not args.sin_ventanas:
            for nombre, posicion, tamano in (("TP5 - camara y AR (mm)", (0, 50), (800, 450)),
                                             ("TP5 - vista externa (mm)", (815, 50), (800, 600))):
                cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(nombre, *tamano)
                cv2.moveWindow(nombre, *posicion)
        i = 0
        while not limite or i < limite:
            ocultar = bool(sim and args.probar_perdida and limite // 3 <= i < limite // 3 + max(3, limite // 10))
            verdad = None
            if sim:
                verdad = sim.generar(i, ocultar_tablero=ocultar)
                imagen, frame_id, tiempo_s = verdad.imagen, verdad.frame_id, verdad.tiempo_s
            else:
                imagen, frame_id, tiempo_s = fuente.leer()
            if video is not None:
                calibracion.validar_imagen(imagen)
                video.write(imagen)
            estado, anotada = procesar_cuadro(
                imagen, calibracion, detector, estimador, frame_id, tiempo_s,
                cartas_config, cuadrado_ar=args.cuadrado_ar,
                renderizador_modelo=renderizador_modelo,
                modelos_por_marcador=modelos_por_marcador)
            pose = estado.camara
            if args.solo_localizacion and pose.valida:
                largo = min(np.linalg.norm(p[1] - p[0]) for p in tablero.marcadores.values()) / 2
                cv2.drawFrameAxes(anotada, calibracion.K, calibracion.dist,
                                 cv2.Rodrigues(pose.T_cw[:3, :3])[0], pose.T_cw[:3, 3], largo, 2)
            if args.solo_localizacion:
                h, w = anotada.shape[:2]
                cv2.rectangle(anotada, (int(.1*w), int(.15*h)),
                              (int(.9*w), int(.85*h)), (150, 150, 150), 1)
                mensaje = "ArUco completo dentro del recuadro | telefono fijo | Q salir"
                for grosor, color in ((3, (0, 0, 0)), (1, (240, 240, 240))):
                    cv2.putText(anotada, mensaje, (15, h-18), cv2.FONT_HERSHEY_SIMPLEX,
                                .48, color, grosor, cv2.LINE_AA)
            if renderizador_modelo is not None:
                h, w = anotada.shape[:2]
                if args.por_marcador:
                    presentes = {i.marcador_id for i in estado.instancias_3d}
                    nombre_modelo = "Modelos por marcador"
                    opciones = " | ".join(f"ID {id}: {modelos_cargados[indice].nombre} "
                                          f"{'visible' if id in presentes else 'oculto'}"
                                          for id, indice in modelos_por_marcador.items())
                    estado_ar = f"{len(presentes)} personajes"
                else:
                    nombre_modelo = renderizador_modelo.modelo_actual.nombre
                    opciones = " | ".join(f"{n+1}: {modelo.nombre}" for n, modelo in enumerate(modelos_cargados))
                    estado_ar = "anclado al marcador" if pose.valida else "oculto: mostrar ArUco"
                mensaje = f"{nombre_modelo} ({args.tamano_modelo_mm:g} mm) | {estado_ar} | {opciones} | Q salir"
                cv2.rectangle(anotada, (8, h-65), (w-8, h-34), (25, 25, 25), -1)
                cv2.putText(anotada, mensaje, (15, h-44), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, (220, 245, 220), 1, cv2.LINE_AA)
            registro = {"frame_id": frame_id, "tiempo_s": tiempo_s, "valida": pose.valida,
                        "ids_usados": list(pose.ids_usados), "error_px": pose.error_px,
                        "motivo": pose.motivo, "antiguedad_s": pose.antiguedad_s,
                        "T_wc": pose.T_wc.tolist() if pose.valida else None,
                        "detecciones": [{"id": d.id, "esquinas_px": d.esquinas.tolist(),
                                         "fuente": getattr(detector, "fuentes", {}).get(d.id, "aruco")}
                                        for d in estado.detecciones],
                        "seguimiento": getattr(detector, "diagnostico", None),
                        "pnp_diagnostico": estimador.diagnostico,
                        "modelo_3d": renderizador_modelo.modelo_actual.nombre if renderizador_modelo is not None and not args.por_marcador else None,
                        "instancias_3d": [{"marcador_id": i.marcador_id, "modelo_indice": i.modelo_indice,
                                          "nombre": modelos_cargados[i.modelo_indice].nombre,
                                          "frame_id": i.frame_id, "T_cm": i.T_cm.tolist(),
                                          "T_wm": None if i.T_wm is None else i.T_wm.tolist()}
                                         for i in estado.instancias_3d],
                        "cartas": [{"id": c.id, "error_px": c.error_px,
                                    "T_wm": c.T_wm.tolist() if c.T_wm is not None else None} for c in estado.cartas]}
            if verdad is not None:
                esperadas = {d.id for d in verdad.detecciones if d.id in cartas_config}
                actuales = {c.id for c in estado.cartas if c.frame_id == frame_id
                            and (not pose.valida or c.T_wm is not None)}
                registro["cartas_faltantes"] = sorted(esperadas - actuales)
                registro["cartas_inesperadas"] = sorted(actuales - esperadas)
                cuadros_cartas_incompletas += int(esperadas != actuales)
                if ocultar:
                    perdidas_esperadas += 1
                    falsas_validas += int(pose.valida)
                if pose.valida:
                    ep, ea = error_transformacion(pose.T_wc, verdad.T_wc)
                    posiciones.append(ep); angulos.append(ea)
                    registro.update(error_posicion_mm=ep, error_orientacion_grados=ea)
                    for carta in estado.cartas:
                        if carta.T_wm is not None and carta.id in verdad.cartas_T_wm:
                            ec, _ = error_transformacion(carta.T_wm, verdad.cartas_T_wm[carta.id])
                            errores_cartas.append(ec)
            if pose.valida:
                if hubo_localizacion and antes_perdida:
                    recuperadas += 1
                hubo_localizacion = True
            antes_perdida = not pose.valida
            registros.append(registro)
            if not visor.actualizar(estado):
                break
            if not args.sin_ventanas:
                cv2.imshow("TP5 - camara y AR (mm)", anotada)
                if args.vista_externa:
                    cv2.imshow("TP5 - vista externa (mm)", visor.renderizar_vista_externa())
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q")):
                    break
                if renderizador_modelo is not None and not args.por_marcador and ord("1") <= tecla < ord("1") + len(modelos_cargados):
                    renderizador_modelo.seleccionar(tecla - ord("1"))
                if args.vista_externa and any(cv2.getWindowProperty(nombre, cv2.WND_PROP_VISIBLE) < 1
                                              for nombre in ("TP5 - camara y AR (mm)", "TP5 - vista externa (mm)")):
                    break
            i += 1
        validos = sum(r["valida"] for r in registros)
        iniciales = next((i for i, r in enumerate(registros) if r["valida"]), len(registros))
        posteriores = len(registros) - iniciales
        resumen = {"fuente": "sintetica" if sim else "webcam", "unidades": "mm",
                   "solo_localizacion": args.solo_localizacion,
                   "seguimiento_continuo": seguimiento_continuo,
                   "modelos_por_marcador": modelos_por_marcador,
                   "lado_mago_mm": lado_mago if args.por_marcador else None,
                   "medida_mago": ("opcion_cli" if args.lado_mago_mm is not None else "predeterminado_verificar_medida") if args.por_marcador else None,
                   "animacion": bool(args.demo_yugioh and not args.sin_animacion),
                   "efectos": bool(args.demo_yugioh and not args.sin_efectos),
                   "modelos_3d": [{"nombre": m.nombre, "ruta": str(m.ruta),
                                   "tamano_mm": m.tamano_mm, "giro_grados": args.giro_modelo_grados,
                                   "atribucion": m.atribucion} for m in modelos_cargados],
                   "cuadrado_ar": {"vertices_mundo_mm": geometria_cuadrado().tolist(),
                                   "requiere_pose_actual": True} if args.cuadrado_ar else None,
                   "opencv": cv2.__version__, "resolucion": list(calibracion.tamano),
                   "cuadros": len(registros), "validos": validos,
                   "perdidos": len(registros) - validos,
                   "cuadros_sin_localizacion_inicial": iniciales,
                   "porcentaje_validos_desde_primera_deteccion": 100 * validos / posteriores if posteriores else None,
                   "video_crudo": str(ruta_video) if ruta_video is not None else None,
                   "video_fps_codificacion": FPS_VIDEO if ruta_video is not None else None,
                   "fps_procesamiento": len(registros) / max(time.monotonic()-inicio, 1e-6),
                   "limites_sinteticos": LIMITES_SINTETICOS if sim else None,
                   "max_error_posicion_mm": max(posiciones, default=None),
                   "max_error_orientacion_grados": max(angulos, default=None),
                   "max_error_carta_mm": max(errores_cartas, default=None),
                   "perdidas_esperadas": perdidas_esperadas,
                   "falsas_validas": falsas_validas, "recuperaciones": recuperadas,
                   "cuadros_cartas_incompletas": cuadros_cartas_incompletas if sim else None}
        if sim:
            resumen["verificacion_ok"] = bool(posiciones and (not cartas_config or errores_cartas)
                and validos == len(registros) - perdidas_esperadas and falsas_validas == 0
                and cuadros_cartas_incompletas == 0
                and max(posiciones) <= LIMITES_SINTETICOS["posicion_mm"]
                and max(angulos) <= LIMITES_SINTETICOS["orientacion_grados"]
                and max(errores_cartas, default=0.) <= LIMITES_SINTETICOS["carta_mm"]
                and (not args.probar_perdida or recuperadas > 0))
        if args.salida:
            args.salida.mkdir(parents=True, exist_ok=True)
            (args.salida / "reporte.json").write_text(json.dumps({"resumen": resumen, "cuadros": registros}, indent=2, ensure_ascii=False), encoding="utf-8")
            if registros:
                if not cv2.imwrite(str(args.salida / "camara_ar.png"), anotada):
                    raise RuntimeError("No se pudo guardar la imagen AR")
                if not cv2.imwrite(str(args.salida / "vista_externa.png"), visor.renderizar_vista_externa()):
                    raise RuntimeError("No se pudo guardar la vista externa")
        print(json.dumps(resumen, indent=2, ensure_ascii=False))
        return 2 if args.verificar_simulacion and not resumen.get("verificacion_ok") else 0
    finally:
        if video is not None:
            video.release()
        if fuente:
            fuente.cerrar()
        if visor:
            visor.cerrar()
        if renderizador_modelo is not None:
            renderizador_modelo.cerrar()
        if not args.sin_ventanas:
            cv2.destroyAllWindows()


def main():
    parser = parser_cli()
    args = parser.parse_args()
    try:
        return ejecutar(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
