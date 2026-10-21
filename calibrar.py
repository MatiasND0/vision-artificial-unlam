#!/usr/bin/env python3
"""Captura y calibra una cámara con un tablero de ajedrez de medidas conocidas."""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibración OpenCV en mm. Usar imágenes crudas a la resolución final de trabajo.")
    sub = parser.add_subparsers(dest="comando", required=True)
    patron = sub.add_parser("patron", help="Generar un SVG imprimible al 100%% y medir sus casilleros")
    imagenes = sub.add_parser("imagenes", help="Calibrar desde imágenes existentes del patrón")
    capturar = sub.add_parser("capturar", help="Capturar vistas con webcam; C guarda, Q calibra, Esc cancela")
    for comando in (patron, imagenes, capturar):
        comando.add_argument("--columnas", type=int, default=9, help="Esquinas interiores horizontales (9)")
        comando.add_argument("--filas", type=int, default=6, help="Esquinas interiores verticales (6)")
        comando.add_argument("--lado-mm", type=float, default=25.0,
                             help="Lado MEDIDO de cada casillero en mm; ejemplo: 25")
        comando.add_argument("--salida", type=Path, default=Path("patron_calibracion.svg") if comando is patron
                             else Path("calibracion.json"))
    imagenes.add_argument("imagenes", nargs="+", help="Rutas o patrones glob; por ejemplo 'capturas/*.png'")
    capturar.add_argument("--camara", default="0", help="Índice de cámara o ruta /dev/videoN")
    capturar.add_argument("--ancho", type=int, default=1280)
    capturar.add_argument("--alto", type=int, default=720)
    capturar.add_argument("--backend", choices=("auto", "v4l2"), default="v4l2" if sys.platform.startswith("linux") else "auto")
    capturar.add_argument("--directorio", type=Path, default=Path("capturas_calibracion"),
                          help="Directorio NUEVO para guardar imágenes crudas y permitir repetir el cálculo")
    return parser


def _capturar(args):
    import cv2
    from tp5.calibracion import MINIMO_VISTAS, calibrar_desde_puntos, detectar_patron, puntos_patron

    if args.ancho <= 0 or args.alto <= 0:
        raise ValueError("Ancho y alto deben ser positivos")
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("La captura interactiva requiere una sesión gráfica; usar imagenes para capturas existentes")
    if args.directorio.exists() and any(args.directorio.iterdir()):
        raise ValueError("El directorio de captura debe estar vacío para no mezclar sesiones")
    args.directorio.mkdir(parents=True, exist_ok=True)
    patron = (args.columnas, args.filas)
    objeto = puntos_patron(patron, args.lado_mm)
    fuente = int(args.camara) if args.camara.isdigit() else args.camara
    backend = cv2.CAP_V4L2 if args.backend == "v4l2" else cv2.CAP_ANY
    camara = cv2.VideoCapture(fuente, backend)
    objetos, imagenes, archivos = [], [], []
    tamano = None
    try:
        if not camara.isOpened():
            raise ValueError(f"No se pudo abrir la cámara {args.camara} con backend {args.backend}")
        camara.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
        camara.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
        print("Mover e inclinar el patrón; variar distancia y recorrer bordes. Mantener enfoque y zoom fijos.")
        print("C: guardar vista válida | Q: calcular calibración | Esc: cancelar sin guardar calibración")
        while True:
            ok, imagen = camara.read()
            if not ok or imagen is None:
                raise ValueError("La cámara dejó de entregar imágenes")
            actual = (imagen.shape[1], imagen.shape[0])
            if tamano is None:
                tamano = actual
                print(f"Resolución efectiva: {actual[0]}×{actual[1]} (se conserva en el archivo)")
            if actual != tamano:
                raise ValueError("La cámara cambió de resolución durante la captura")
            esquinas = detectar_patron(imagen, patron)
            vista = imagen.copy()
            if esquinas is not None:
                cv2.drawChessboardCorners(vista, patron, esquinas.reshape(-1, 1, 2), True)
            mensaje = f"Vistas {len(imagenes)} (min {MINIMO_VISTAS}, recomendado 15-25) | C guardar / Q calibrar / Esc salir"
            cv2.putText(vista, mensaje, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 230, 0) if esquinas is not None else (0, 100, 255), 1, cv2.LINE_AA)
            cv2.imshow("TP5 calibracion - imagen cruda", vista)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                print("Captura cancelada; las imágenes guardadas siguen disponibles.")
                return None
            if tecla in (ord("c"), ord("C")):
                if esquinas is None:
                    print("No se ve el patrón completo: vista descartada.")
                    continue
                ruta = args.directorio / f"vista_{len(imagenes):03d}.png"
                if not cv2.imwrite(str(ruta), imagen):
                    raise OSError(f"No se pudo guardar {ruta}")
                objetos.append(objeto.copy())
                imagenes.append(esquinas)
                archivos.append(str(ruta))
                print(f"Guardada {ruta}")
            if tecla in (ord("q"), ord("Q")):
                if len(imagenes) < MINIMO_VISTAS:
                    print(f"Faltan vistas: se requieren al menos {MINIMO_VISTAS}.")
                    continue
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()
    return calibrar_desde_puntos(objetos, imagenes, tamano, {
        "origen": "captura_webcam", "camara": args.camara, "backend": args.backend,
        "patron": "chessboard", "esquinas_interiores": list(patron),
        "lado_casillero_mm": args.lado_mm, "imagenes_aceptadas": archivos,
    })


def main(argv=None) -> int:
    parser = crear_parser()
    args = parser.parse_args(argv)
    try:
        from tp5.calibracion import (calibrar_desde_imagenes, guardar_calibracion,
                                     guardar_patron_svg, validar_patron)
        validar_patron((args.columnas, args.filas), args.lado_mm)
        if args.comando == "patron":
            guardar_patron_svg(args.salida, (args.columnas, args.filas), args.lado_mm)
            print(f"Patrón guardado en {args.salida}. Imprimir al 100%, montar plano y medir un casillero.")
            return 0
        if args.comando == "imagenes":
            rutas = sorted({ruta for entrada in args.imagenes for ruta in (glob.glob(entrada) or [entrada])})
            calibracion = calibrar_desde_imagenes(rutas, (args.columnas, args.filas), args.lado_mm)
        else:
            calibracion = _capturar(args)
            if calibracion is None:
                return 0
        guardar_calibracion(calibracion, args.salida)
        print(f"Calibración guardada en {args.salida}: {calibracion.tamano[0]}×{calibracion.tamano[1]}, "
              f"{calibracion.metadatos['vistas']} vistas, RMS {calibracion.rms_px:.4f} px.")
        print("Verificar la reproyección con la cámara y conservar esta resolución, enfoque y zoom.")
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except ImportError as error:
        parser.error(f"Falta una dependencia: {error}. Instalar requirements.txt en el entorno virtual.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
