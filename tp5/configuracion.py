"""Configuración inicial de EJEMPLO; medir antes de usar con cámara real."""
from .modelos import CartaConfig

DICCIONARIO = "4x4_50"
IDS_TABLERO = (0, 1, 2, 3)
LADO_TABLERO_MM = 60.0
TAMANO_TABLERO_MM = (600.0, 400.0)
# Origen en el centro del tablero; X derecha, Y fondo, Z arriba.
CENTROS_TABLERO_MM = {0: (-250.0, -150.0), 1: (250.0, -150.0),
                       2: (250.0, 150.0), 3: (-250.0, 150.0)}
CARTAS = {
    10: CartaConfig(10, "Mago Oscuro", 40.0, 2500, 2100, (220, 80, 200)),
    11: CartaConfig(11, "Dragón Blanco de Ojos Azules", 40.0, 3000, 2500, (240, 220, 80)),
}
TAMANO_IMAGEN = (1280, 720)
ERROR_MAX_PX = 3.0
