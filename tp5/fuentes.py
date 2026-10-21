"""Captura cruda sin corregir distorsión ni modificar resolución después de leer."""
import time

import cv2


class FuenteCamara:
    def __init__(self, indice=0, tamano=(1280, 720)):
        # CAP_ANY permite elegir un backend disponible en Linux/Windows/macOS.
        self.captura = cv2.VideoCapture(indice, cv2.CAP_ANY)
        if not self.captura.isOpened():
            self.captura.release()
            raise RuntimeError(f"No se pudo abrir la cámara {indice}")
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, tamano[0])
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, tamano[1])
        self.frame_id = 0

    def leer(self):
        ok, imagen = self.captura.read()
        if not ok or imagen is None:
            raise RuntimeError("La cámara dejó de entregar imágenes; no hay una medición actual")
        frame_id = self.frame_id
        self.frame_id += 1
        return imagen, frame_id, time.monotonic()

    def cerrar(self):
        self.captura.release()
