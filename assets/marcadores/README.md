# Dos personajes con marcadores propios

- `aruco_id0_dragon.png`: referencia fija ID 0, Dragón Blanco.
- `aruco_id10_mago.png`: marcador móvil ID 10, Mago Oscuro.
- `dos_marcadores.png`: lámina alternativa con ambos; medirlos nuevamente en pantalla.

Diccionario `4x4_50`. Las imágenes individuales son de 2400×1800 píxeles,
con cuadrado negro de 1200×1200 en idéntica posición. El ID 0 original fue
medido en 32 mm sobre el iPhone. Mostrar el ID 10 individual con la misma
escala y comprobar que su lado negro mida también 32 mm; si cambia, indicar
la medida con `--lado-mago-mm`. El código no registra como medida física nueva
el valor predeterminado.

La lámina doble tiene otra distribución: no hereda la escala de las imágenes
individuales. Si se muestra en un teléfono, medir también el ID 0 y actualizar
sus cuatro esquinas en el tablero. Para evitar cambiar el montaje ya calibrado,
se pueden alternar las dos imágenes individuales o usar otro teléfono/pantalla
para el Mago, midiendo su marcador. Con ambos visibles se renderizan a la vez.

Mantener fijo el ID 0 mientras se mueve la notebook. El ID 10 puede moverse:
su pose nunca se usa para localizar la cámara en el mundo. Si solo está visible
el ID 10, el Mago se ve relativo a cámara y la localización mundial se declara
perdida. Al ocultar uno, solo desaparece su personaje.

```bash
.venv/bin/python main.py --por-marcador --vista-externa \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --lado-mago-mm 32 --camara 0 --salida salidas/demo_dos_marcadores
```

Este modo conserva la animación y los efectos. No hace falta seleccionar 1/2;
cada ID elige su personaje. Q o Escape terminan la prueba.
