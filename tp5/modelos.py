"""Contratos compartidos. T_ab transforma puntos de b a a; longitudes en mm."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Calibracion:
    K: np.ndarray
    dist: np.ndarray
    tamano: tuple[int, int]
    rms_px: float = 0.0
    metadatos: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).copy()
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1).copy()
        self.tamano = tuple(self.tamano)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("K debe ser una matriz finita 3×3")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError("Intrínsecos inválidos")
        if self.dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.dist).all():
            raise ValueError("Distorsión OpenCV inválida")
        if len(self.tamano) != 2 or any(int(v) != v or v <= 0 for v in self.tamano):
            raise ValueError("Resolución inválida (ancho, alto)")
        if not np.isfinite(self.rms_px) or self.rms_px < 0:
            raise ValueError("RMS inválido")

    def validar_imagen(self, imagen: np.ndarray) -> None:
        if (imagen.shape[1], imagen.shape[0]) != self.tamano:
            raise ValueError(f"Resolución {imagen.shape[1]}×{imagen.shape[0]} incompatible con calibración {self.tamano}")


@dataclass
class Deteccion:
    id: int
    esquinas: np.ndarray  # (4, 2), orden canónico ArUco TL, TR, BR, BL

    def __post_init__(self):
        self.esquinas = np.asarray(self.esquinas, dtype=np.float64).reshape(4, 2).copy()
        if not np.isfinite(self.esquinas).all():
            raise ValueError("Esquinas no finitas")


@dataclass
class Tablero:
    diccionario: str
    marcadores: dict[int, np.ndarray]  # esquinas (4,3) en mundo, mm
    tamano_mm: tuple[float, float]
    es_ejemplo: bool = True

    def __post_init__(self):
        self.marcadores = {int(i): np.asarray(p, dtype=np.float64).reshape(4, 3).copy()
                           for i, p in self.marcadores.items()}
        if not self.marcadores or any(not np.isfinite(p).all() for p in self.marcadores.values()):
            raise ValueError("Tablero vacío o geometría no finita")


@dataclass(frozen=True)
class CartaConfig:
    id: int
    nombre: str
    lado_mm: float
    ataque: int
    defensa: int
    color_bgr: tuple[int, int, int] = (0, 200, 255)


@dataclass
class PoseCamara:
    frame_id: int
    tiempo_s: float
    valida: bool
    T_cw: np.ndarray | None = None
    T_wc: np.ndarray | None = None
    ids_usados: tuple[int, ...] = ()
    error_px: float | None = None
    motivo: str = ""
    ultima_T_wc: np.ndarray | None = None
    antiguedad_s: float | None = None

    def __post_init__(self):
        if self.valida and (self.T_cw is None or self.T_wc is None):
            raise ValueError("Una pose válida requiere ambas transformaciones")
        if not self.valida and (self.T_cw is not None or self.T_wc is not None):
            raise ValueError("Una pose perdida no tiene transformaciones actuales")


@dataclass
class PoseCarta:
    id: int
    frame_id: int
    T_cm: np.ndarray
    error_px: float
    T_wm: np.ndarray | None = None  # solo con cámara válida del MISMO cuadro


@dataclass
class EstadoMundo:
    camara: PoseCamara
    cartas: list[PoseCarta] = field(default_factory=list)
    detecciones: list[Deteccion] = field(default_factory=list)  # observaciones del mismo cuadro


@dataclass
class CuadroSintetico:
    imagen: np.ndarray
    frame_id: int
    tiempo_s: float
    T_cw: np.ndarray
    T_wc: np.ndarray
    cartas_T_wm: dict[int, np.ndarray]
    detecciones: list[Deteccion]  # esquinas exactas para pruebas geométricas
