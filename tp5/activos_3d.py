"""Carga estática de GLB para AR: materiales, jerarquía y skin en milímetros.

glTF usa Y hacia arriba; la salida usa Z hacia arriba y UV con (0, 0) en la
esquina superior izquierda de la textura. Las animaciones no se reproducen:
se conservan las transformaciones originales de los nodos y se hornea su skin.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import struct

import numpy as np


@dataclass
class Malla3D:
    vertices_mm: np.ndarray
    normales: np.ndarray
    uv: np.ndarray
    triangulos: np.ndarray
    textura_rgba: np.ndarray
    color_rgba: np.ndarray
    alpha_modo: str = "OPAQUE"
    alpha_corte: float = .5


@dataclass
class Modelo3D:
    ruta: Path
    nombre: str
    partes: list[Malla3D]
    tamano_mm: float
    atribucion: dict


def _cabecera_glb(datos: bytes) -> tuple[dict, bytes]:
    """Lee solo JSON y conserva intactos los bloques binarios para trimesh."""
    if len(datos) < 20:
        raise ValueError("El archivo no contiene una cabecera GLB completa.")
    magic, version, total, longitud, tipo = struct.unpack_from("<5I", datos)
    if magic != 0x46546C67 or version != 2 or total != len(datos):
        raise ValueError("Se necesita un archivo GLB versión 2 completo.")
    if tipo != 0x4E4F534A or 20 + longitud > total:
        raise ValueError("Falta el bloque JSON del archivo GLB.")
    try:
        cabecera = json.loads(datos[20:20 + longitud])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("El bloque JSON del GLB no es válido.") from exc
    if not isinstance(cabecera, dict):
        raise ValueError("El bloque JSON del GLB debe ser un objeto.")
    return cabecera, datos[20 + longitud:]


def _preparar_skin(cabecera: dict, cola: bytes) -> bytes:
    """Expone accessors de skin a trimesh, sin interpretar buffers propios.

    trimesh 4.x no interpreta skin. Sus atributos personalizados sí conservan
    accessors completos, incluidos MAT4; aliasamos esos atributos en una copia
    en memoria. Nombres deterministas permiten consultar el grafo sin ambigüedad.
    No se modifica el GLB original ni se evalúan sus canales de animación.
    """
    for indice, nodo in enumerate(cabecera.get("nodes", [])):
        nodo["name"] = f"_TP5_node_{indice}"
    for malla in cabecera.get("meshes", []):
        for primitiva in malla.get("primitives", []):
            atributos = primitiva["attributes"]
            for clave, valor in list(atributos.items()):
                if clave.startswith(("JOINTS_", "WEIGHTS_")):
                    atributos["_TP5_" + clave] = valor
            for indice, skin in enumerate(cabecera["skins"]):
                if "inverseBindMatrices" in skin:
                    atributos[f"_TP5_BIND_{indice}"] = skin["inverseBindMatrices"]
    texto = json.dumps(cabecera, ensure_ascii=False).encode("utf-8")
    texto += b" " * (-len(texto) % 4)
    return (struct.pack("<5I", 0x46546C67, 2, 20 + len(texto) + len(cola),
                        len(texto), 0x4E4F534A) + texto + cola)


def _unitarias(normales: np.ndarray) -> np.ndarray:
    longitud = np.linalg.norm(normales, axis=1)
    if not np.isfinite(normales).all() or np.any(longitud < 1e-12):
        raise ValueError("El modelo contiene normales inválidas o degeneradas.")
    return normales / longitud[:, None]


def _normales_malla(vertices: np.ndarray, caras: np.ndarray) -> np.ndarray:
    """Alternativa NumPy a vertex_normals, sin dependencia opcional de SciPy."""
    puntos = vertices[caras]
    normales_caras = np.cross(puntos[:, 1] - puntos[:, 0],
                              puntos[:, 2] - puntos[:, 0])
    normales = np.zeros_like(vertices, dtype=np.float64)
    for columna in range(3):
        np.add.at(normales, caras[:, columna], normales_caras)
    return _unitarias(normales)


def _deformar_skin(malla, indices, escena, skin: dict, indice_skin: int):
    """Hornea worldJoint @ inverseBind usando los pesos de cada vértice."""
    atributos = malla.vertex_attributes
    juntas = skin.get("joints", [])
    if not juntas:
        raise ValueError("El skin no contiene articulaciones.")
    clave_bind = f"_TP5_BIND_{indice_skin}"
    if clave_bind in atributos:
        # glTF almacena MAT4 por columnas; trimesh decodifica por filas.
        inversas = np.asarray(atributos[clave_bind], dtype=float).transpose(0, 2, 1)
    else:
        inversas = np.tile(np.eye(4), (len(juntas), 1, 1))
    if inversas.shape != (len(juntas), 4, 4) or not np.isfinite(inversas).all():
        raise ValueError("Las matrices inversas del skin no son válidas.")
    globales = np.stack([escena.graph[f"_TP5_node_{j}"][0] for j in juntas])
    paleta = globales @ inversas
    mezcla = np.zeros((len(indices), 4, 4), dtype=float)
    suma = np.zeros(len(indices), dtype=float)
    grupos = sorted(k for k in atributos if k.startswith("_TP5_JOINTS_"))
    if not grupos:
        raise ValueError("La malla tiene skin pero no contiene JOINTS/WEIGHTS.")
    for clave in grupos:
        clave_pesos = clave.replace("JOINTS", "WEIGHTS")
        if clave_pesos not in atributos:
            raise ValueError("Faltan pesos para las articulaciones del skin.")
        ids = np.asarray(atributos[clave])[indices]
        pesos_originales = np.asarray(atributos[clave_pesos])
        pesos = pesos_originales[indices].astype(float)
        if pesos_originales.dtype.kind in "ui":
            pesos /= np.iinfo(pesos_originales.dtype).max
        if (ids.ndim != 2 or ids.shape != pesos.shape or ids.dtype.kind not in "ui"
                or np.any(ids >= len(juntas)) or np.any(pesos < 0)
                or not np.isfinite(pesos).all()):
            raise ValueError("Los índices o pesos del skin no son válidos.")
        mezcla += np.einsum("nk,nkij->nij", pesos, paleta[ids])
        suma += pesos.sum(axis=1)
    if np.any(suma < 1e-12):
        raise ValueError("El skin contiene vértices sin influencia válida.")
    mezcla /= suma[:, None, None]
    return mezcla


def _material(malla, indices):
    visual = malla.visual
    material = getattr(visual, "material", None)
    textura = getattr(material, "baseColorTexture", None)
    if textura is None:
        textura = getattr(material, "image", None)
    if textura is None:
        rgba = np.full((1, 1, 4), 255, dtype=np.uint8)
    else:
        # PIL mantiene la primera fila arriba: no invertir la imagen aquí.
        rgba = np.array(textura.convert("RGBA"), dtype=np.uint8)
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = getattr(material, "diffuse", None)
    if color is None and getattr(visual, "kind", None) in ("vertex", "face"):
        colores = (visual.vertex_colors[indices] if visual.kind == "vertex"
                   else visual.face_colors)
        if len(colores) and not np.all(colores == colores[0]):
            raise ValueError("Se necesitan materiales o texturas para colores variables.")
        color = colores[0] if len(colores) else None
    if color is None:
        color = np.ones(4, dtype=np.float32)
    else:
        original = np.asarray(color)
        color = original.astype(np.float32)
        if original.dtype.kind in "ui":
            color /= 255.0
        if color.shape == (3,):
            color = np.r_[color, np.float32(1)]
    if color.shape != (4,) or not np.isfinite(color).all():
        raise ValueError("El material contiene un color inválido.")
    color = np.clip(color, 0, 1).astype(np.float32)
    uv_original = getattr(visual, "uv", None)
    if uv_original is None:
        if textura is not None:
            raise ValueError("El material tiene textura pero no coordenadas UV.")
        uv = np.zeros((len(indices), 2), dtype=np.float32)
    else:
        uv = np.asarray(uv_original, dtype=np.float32)[indices].copy()
        # trimesh pasa glTF a UV inferiores; el renderer recibe UV superiores.
        uv[:, 1] = 1 - uv[:, 1]
    if uv.shape != (len(indices), 2) or not np.isfinite(uv).all():
        raise ValueError("El modelo contiene coordenadas UV inválidas.")
    modo = str(getattr(material, "alphaMode", None) or "OPAQUE").upper()
    corte = getattr(material, "alphaCutoff", None)
    corte = .5 if corte is None else float(corte)
    if modo not in ("OPAQUE", "MASK", "BLEND") or not np.isfinite(corte):
        raise ValueError("El material contiene una transparencia inválida.")
    return rgba, color, uv, modo, corte


def cargar_modelo(ruta, tamano_mm=90.0, giro_grados=0.0) -> Modelo3D:
    """Carga un GLB y escala su extensión máxima al tamaño pedido.

    El conjunto se centra en XY y apoya su punto mínimo en Z=2 mm. El giro
    opcional es alrededor de Z, antes de calcular sus límites y normalización.
    Texturas y color son baseColor de glTF; no se evalúan luces, animaciones,
    morfismos ni mapas PBR de rugosidad/metal. No descarga recursos remotos.
    """
    tamano_mm, giro_grados = float(tamano_mm), float(giro_grados)
    if not np.isfinite(tamano_mm) or tamano_mm <= 0:
        raise ValueError("El tamaño del modelo debe ser positivo y finito.")
    if not np.isfinite(giro_grados):
        raise ValueError("El giro del modelo debe ser finito.")
    ruta = Path(ruta).expanduser().resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta}")
    if ruta.suffix.lower() != ".glb":
        raise ValueError("Se necesita un modelo en formato .glb.")
    cabecera, cola = _cabecera_glb(ruta.read_bytes())
    atribucion = dict(cabecera.get("asset", {}).get("extras", {}))
    nombre = str(atribucion.get("title") or ruta.stem)
    skins = cabecera.get("skins", [])
    animaciones = len(cabecera.get("animations", []))
    # Importación diferida: el resto del TP no requiere dependencias 3D.
    import trimesh
    if skins:
        datos = _preparar_skin(cabecera, cola)
        escena = trimesh.load_scene(io.BytesIO(datos), file_type="glb", process=False)
    else:
        escena = trimesh.load_scene(ruta, process=False)
    a = np.radians(giro_grados)
    rotacion = np.array([[np.cos(a), -np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0], [0, 0, 1.]])
    # Rx(+90 grados): Y glTF -> Z mundo; Z glTF -> -Y mundo.
    rotacion = rotacion @ np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    partes = []
    skins_aplicadas = 0
    for nodo in escena.graph.nodes_geometry:
        transformacion, geometria = escena.graph[nodo]
        malla = escena.geometry[geometria]
        if not isinstance(malla, trimesh.Trimesh) or len(malla.faces) == 0:
            continue
        caras_originales = np.asarray(malla.faces)
        indices, caras = np.unique(caras_originales, return_inverse=True)
        caras = caras.reshape(-1, 3)
        vertices = np.asarray(malla.vertices, dtype=float)[indices]
        if not np.isfinite(vertices).all():
            raise ValueError("El modelo contiene posiciones no finitas.")
        # Usar normales glTF ya cargadas. Calcularlas con NumPy si no existen.
        normales_cargadas = malla._cache.cache.get("vertex_normals")
        normales = (_normales_malla(vertices, caras) if normales_cargadas is None
                    else np.asarray(normales_cargadas, dtype=float)[indices])
        indice_skin = None
        if skins:
            indice_nodo = int(nodo.removeprefix("_TP5_node_").split("_")[0])
            indice_skin = cabecera["nodes"][indice_nodo].get("skin")
        try:
            if indice_skin is None:
                vertices = vertices @ transformacion[:3, :3].T + transformacion[:3, 3]
                normales = normales @ np.linalg.inv(transformacion[:3, :3])
            else:
                mezcla = _deformar_skin(malla, indices, escena,
                                        skins[indice_skin], indice_skin)
                vertices = (np.einsum("nij,nj->ni", mezcla[:, :3, :3], vertices)
                            + mezcla[:, :3, 3])
                inversas = np.linalg.inv(mezcla[:, :3, :3])
                normales = np.einsum("ni,nij->nj", normales, inversas)
                skins_aplicadas += 1
        except np.linalg.LinAlgError as exc:
            raise ValueError("El modelo contiene una transformación singular.") from exc
        vertices = vertices @ rotacion.T
        normales = _unitarias(normales @ rotacion.T)
        textura, color, uv, modo, corte = _material(malla, indices)
        partes.append(Malla3D(vertices, normales.astype(np.float32), uv,
                              caras.astype(np.uint32), textura, color, modo, corte))
    if not partes:
        raise ValueError("El GLB no contiene ninguna malla triangular visible.")
    todos = np.concatenate([p.vertices_mm for p in partes])
    minimo, maximo = todos.min(axis=0), todos.max(axis=0)
    extension = float(np.max(maximo - minimo))
    if not np.isfinite(extension) or extension < 1e-12:
        raise ValueError("El modelo no tiene dimensiones válidas.")
    escala = tamano_mm / extension
    centro = np.r_[(minimo[:2] + maximo[:2]) / 2, minimo[2]]
    for parte in partes:
        parte.vertices_mm = ((parte.vertices_mm - centro) * escala
                             + [0., 0., 2.]).astype(np.float32)
    atribucion.update(pose_estatica="transformaciones originales de nodos",
                      skins_aplicadas=skins_aplicadas,
                      animaciones_ignoradas=animaciones)
    return Modelo3D(ruta, nombre, partes, tamano_mm, atribucion)
