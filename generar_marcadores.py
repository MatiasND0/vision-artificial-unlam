"""Genera ArUco a escala física: imprimir SVG al 100 %, medir antes de montar.

La generación por módulos SVG y regla de control se adapta de
``tp3-localizacion-homografica/generar_marcadores.py``. Un marcador por hoja evita
recortes al imprimir. Los tamaños predeterminados son EJEMPLOS, en mm.
"""
import argparse
from pathlib import Path

import cv2

from tp5.configuracion import CARTAS, DICCIONARIO, IDS_TABLERO, LADO_TABLERO_MM
from tp5.deteccion import Detector


def generar_svg(id_marcador, lado_mm, diccionario=DICCIONARIO):
    if not 0 < lado_mm <= 160:
        raise ValueError("Para la hoja A4 el lado debe estar entre 0 y 160 mm")
    detector = Detector(diccionario)
    n = detector.diccionario.markerSize + 2
    bits = detector.generar(id_marcador, n) < 128
    x, y = (210 - lado_mm) / 2, 65.
    modulo = lado_mm / n
    elementos = ['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
                 '<rect width="210" height="297" fill="white"/>',
                 '<g font-family="sans-serif" font-size="4" fill="black">',
                 f'<text x="15" y="20">ArUco {diccionario} · ID {id_marcador} · lado negro {lado_mm:g} mm</text>',
                 '<text x="15" y="28">EJEMPLO: imprimir al 100 %, sin ajustar a página. Medir.</text>',
                 '<text x="70" y="45">Borde superior hacia +Y del mundo</text>',
                 '<path d="M105 59 L105 48 M102 52 L105 48 L108 52" stroke="black" fill="none"/>']
    for fila in range(n):
        for col in range(n):
            if bits[fila, col]:
                elementos.append(f'<rect x="{x + col * modulo:.5f}" y="{y + fila * modulo:.5f}" '
                                 f'width="{modulo + .001:.5f}" height="{modulo + .001:.5f}"/>')
    margen = max(8., lado_mm / 6)
    elementos.append(f'<rect x="{x - margen:.4f}" y="{y - margen:.4f}" '
                     f'width="{lado_mm + 2 * margen:.4f}" height="{lado_mm + 2 * margen:.4f}" '
                     'fill="none" stroke="#aaa" stroke-width="0.15" stroke-dasharray="2 2"/>')
    elementos.extend([f'<text x="15" y="250">Conservar margen blanco ≥ {margen:g} mm. No rotar el montaje.</text>',
                      '<line x1="20" y1="270" x2="120" y2="270" stroke="black" stroke-width="0.3"/>',
                      '<line x1="20" y1="266" x2="20" y2="274" stroke="black" stroke-width="0.3"/>',
                      '<line x1="120" y1="266" x2="120" y2="274" stroke="black" stroke-width="0.3"/>',
                      '<text x="20" y="281">Control: esta línea debe medir 100 mm.</text>', '</g></svg>'])
    return "\n".join(elementos) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=int, nargs="+", default=[*IDS_TABLERO, *CARTAS])
    parser.add_argument("--lado", type=float, help="Lado negro común en mm; omitir para 60 mm tablero / 40 mm cartas")
    parser.add_argument("--diccionario", default=DICCIONARIO)
    parser.add_argument("--salida", type=Path, default=Path("marcadores"))
    args = parser.parse_args()
    detector = Detector(args.diccionario)
    args.salida.mkdir(parents=True, exist_ok=True)
    for identificador in args.ids:
        lado = args.lado if args.lado is not None else (CARTAS[identificador].lado_mm if identificador in CARTAS else LADO_TABLERO_MM)
        svg = args.salida / f"aruco_{args.diccionario}_id{identificador}_{lado:g}mm.svg"
        svg.write_text(generar_svg(identificador, lado, args.diccionario), encoding="utf-8")
        png = svg.with_suffix(".png")
        if not cv2.imwrite(str(png), detector.generar(identificador, 600, borde_px=120)):
            raise OSError(f"No se pudo guardar {png}")
        print(f"ID {identificador}: {svg} (lado negro {lado:g} mm) y {png}")
    print("Imprimir SVG al 100 %, verificar regla y medir el lado negro. PNG solo como referencia.")
    print("Montaje tablero: borde superior del marcador hacia +Y, normal visible hacia +Z.")
    print("Los IDs 10 y 11 son cartas móviles; no usarlos como referencias fijas.")


if __name__ == "__main__":
    main()
