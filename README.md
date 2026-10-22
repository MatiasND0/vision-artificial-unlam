# TP5 — Mesa de duelo AR

Localización de una cámara móvil respecto de un tablero ArUco fijo, cartas
independientes y geometría AR. Todas las longitudes están en **milímetros**.
El tablero grande y las cartas incluidos siguen siendo **ejemplos sintéticos**.
La webcam ya está calibrada y la referencia única del iPhone está medida en 32 mm.

## Calibración real disponible

`calibraciones/webcam.json`: webcam 0, resolución 1280×720, 25 capturas del damero
mostrado en un iPhone y casilleros de 7,4 mm (10 casillas medidas: 74 mm).
RMS de ajuste: **0,2783 px**; peor captura: **0,4478 px**. Se redetectaron las
54 esquinas en las 25 imágenes y se reprodujo el RMS guardado. En validación
cruzada de cinco particiones, el máximo RMS de una imagen reservada fue
**0,4535 px**. El informe está en `salidas/revision_calibracion_iphone.json`.

Esto permite iniciar las pruebas con ArUco; no certifica precisión métrica. La
zona superior y el extremo derecho de la imagen tienen poca cobertura del
patrón. Antes de la validación final conviene añadir vistas en esas zonas y
comprobar un marcador de tamaño medido. Mantener resolución, enfoque y zoom.
El marcador `salidas/aruco_iphone_id0.png` ya fue medido por el usuario en la
pantalla: cuadrado negro de **32 mm**. Esa medida se guardó en `tablero_iphone.json`.

## Primer objeto AR: cuadrado sobre el iPhone

No requiere cartas ni un modelo 3D. Con el mismo ArUco ID 0 de 32 mm y su
zoom sin cambios, dejar fijo el iPhone y ejecutar desde el proyecto:

```bash
.venv/bin/python main.py --solo-localizacion --cuadrado-ar --vista-externa \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --camara 0 --salida salidas/prueba_cuadrado
```

Un cuadrado de **40 × 40 mm**, con relleno translúcido, queda centrado frente
al marcador y separado **20 mm de la pantalla**, hacia su cara visible (+Z).
Es paralelo a la pantalla; al mover lentamente la notebook cambia su perspectiva
manteniendo el anclaje. No cambiar el zoom del marcador medido. Mantenerlo entero
dentro del recuadro de guía y con margen blanco visible.

Si se pierde la referencia, el cuadrado desaparece y vuelve con una pose actual
válida; no se mantiene pegado a una posición vieja de la imagen. Q o Escape
cierran la demostración y guardan imágenes y reporte. `--guardar-video` permite
conservar además la captura cruda; en ese caso usar una carpeta de salida nueva.
La segunda ventana sigue mostrando la localización de la cámara en el mundo.

## Modelos 3D: Dragón Blanco y Mago Oscuro

Los dos GLB descargados por el usuario están en `assets/modelos`, con sus
texturas y créditos originales. Instalar las dependencias opcionales de AR 3D
en el mismo entorno (ya instaladas y con EGL comprobado en esta notebook):

```bash
.venv/bin/python -m pip install -r requirements-ar3d.txt
.venv/bin/python main.py --solo-localizacion --demo-yugioh --vista-externa \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --camara 0 --salida salidas/demo_yugioh
```

Mostrar el mismo ArUco ID 0 de 32 mm, sin cambiar su zoom, y mantener fijo el
teléfono. Se inicia con el Dragón Blanco; con la ventana de cámara enfocada,
**1** selecciona el dragón y **2** el Mago Oscuro. **Q** o Escape cierran la demo.
Cada personaje mantiene su base frente al centro del marcador. Al perder la
referencia se oculta y vuelve cuando se recupera una pose actual válida.

La mayor dimensión del personaje mide 90 mm; `--tamano-modelo-mm 65` cambia
ese tamaño, conservando proporciones. `--giro-modelo-grados 180` permite orientar
el frente alrededor del eje Z del marcador. La base queda a 2 mm de su plano.
Para otros archivos, usar `--modelo-3d ruta/modelo.glb` en lugar de
`--demo-yugioh`; repetir la opción para alternar hasta nueve modelos.

La demo incluye aleteo procedural y flotación suave del dragón, respiración del
Mago Oscuro, círculo de invocación y partículas. El dragón descargado **no tiene
esqueleto ni clips**: se flexiona la zona de sus alas, respetando cuerpo y raíz.
Los brazos mantienen la pose base del archivo; no se creó un rig completo.
El Mago Oscuro sí trae skin, horneado en la pose de sus nodos; su clip original
no se reproduce. `--sin-animacion` conserva la pose estática y `--sin-efectos`
oculta el círculo y las partículas. Todo se oculta al perder una pose actual.

El render usa ModernGL/EGL con profundidad, texturas y luz simple. La geometría
se ajusta de glTF (Y arriba) al mundo del TP (Z arriba). La imagen de cámara
permanece cruda; se aplica distorsión al overlay una sola vez para coincidir
con los mismos K/dist de PnP. No se calcula oclusión por manos u objetos reales.
Esta opción requiere OpenGL 3.3 con EGL; funciona aquí con Mesa Intel y no
requiere instalar Open3D. Los créditos están en `assets/modelos/README.md` y
se conservan también en el reporte de cada sesión.

## Un marcador para cada personaje

`--por-marcador` activa **ID 0 → Dragón Blanco** e **ID 10 → Mago Oscuro**.
Pueden aparecer juntos y moverse de forma independiente. Las imágenes están
en `assets/marcadores`; el ID 10 individual tiene el mismo lienzo y tamaño en
píxeles que el ID 0 que se midió en el iPhone. Conservar su escala y comprobar
que el lado negro del nuevo marcador mida **32 mm**, o indicar la medida real:

```bash
.venv/bin/python main.py --por-marcador --vista-externa \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --lado-mago-mm 32 --camara 0 --salida salidas/demo_dos_marcadores
```

El ID 0 permanece como referencia fija para la cámara. El Mago sigue al ID 10,
que puede moverse y no localiza la cámara. Si el ID 0 desaparece pero el ID 10
sigue visible, el Mago permanece visible respecto de la cámara y se indica que
falta la referencia mundial. Cada personaje usa su pose actual; no se conserva
la del personaje oculto. Los modelos comparten el buffer de profundidad.

Se puede alternar entre las imágenes individuales en un iPhone o mostrar ambas
en soportes distintos. La lámina `dos_marcadores.png` permite mostrarlas juntas,
pero **requiere medir nuevamente ambos cuadrados**, porque cambia su escala de
visualización. Su tamaño no se supone igual al de las imágenes individuales.
El modo anterior con las teclas 1/2 sigue disponible con `--demo-yugioh`.

## Combate visual entre los personajes

`--combate` incluye el modo de dos marcadores. Al ver **ID 0 e ID 10** de forma
continua durante 0,25 s, los personajes miran hacia el rival y alternan ataques:
rayo cian del Dragón y orbe violeta del Mago, con carga, impacto, pequeños
avances y retrocesos. Ambos comienzan con 100 de vida; el Dragón inflige 25 y
el Mago 20 por impacto. Es una demostración visual simplificada, sin las reglas
completas del juego de cartas.

```bash
.venv/bin/python main.py --combate --vista-externa \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --lado-mago-mm 32 --camara 0 --salida salidas/demo_combate
```

**Espacio** pausa o continúa, **R** reinicia vidas y turnos, **Q** o Escape salen.
Si falta un marcador, el combate pausa su reloj y no aplica daño; el personaje
que siga observado puede verse en su ancla actual. Al recuperar ambos se
comprueba otra vez la presencia antes de continuar. No se completan disparos
con una pose guardada ni se avanza la pelea durante una interrupción larga.
Con un solo iPhone alternando imágenes pueden verse los personajes por separado;
para que peleen, ambos marcadores deben estar visibles al mismo tiempo.

Las orientaciones y movimientos de pelea solo afectan las poses enviadas al
render: no modifican PnP, las coordenadas medidas, el tablero ni los GLB. El
reporte conserva las anclas originales y el estado de combate por cuadro,
incluidas vidas, fase y puntos de ataque en la cámara actual. Las luces usan
sprites proyectados en dos capas, con K/dist una sola vez: ataques detrás de
los modelos e impactos delante. No calculan oclusión por objetos reales.

## Seguimiento continuo de la referencia

La webcam usa por defecto el detector ArUco, una segunda búsqueda localizada
en la imagen actual y flujo óptico verificado para pérdidas breves. El flujo
caduca a los **0,3 s** sin una nueva decodificación; exige seguimiento de
ida/vuelta, homografía consistente y coincidencia del patrón binario del
marcador en la imagen nueva. No congela la pose ni promedia esquinas entre
cuadros. En el modo de dos personajes también se recupera visualmente el ID 10,
sin agregarlo a las referencias fijas usadas por PnP de cámara.

PnP conserva los límites físicos y de reproyección. Para intervalos cortos
también comprueba la continuidad de las ramas; si todas muestran un salto
incompatible, declara pérdida en lugar de reutilizar la pose. Después de una
ausencia larga permite relocalizar. `--seguimiento-basico` permite comparar
con el detector y estimador anteriores. La simulación conserva el modo base.
El reporte distingue `aruco`, `aruco_roi` y `flujo_verificado`, y conserva los
candidatos PnP y los rechazos por continuidad.

Sobre los mismos videos MJPEG disponibles, sin medir exactitud física:

| Grabación | Cuadros con pose: base → continuo | Apariciones separadas: base → continuo |
| --- | --- | --- |
| Segunda prueba, 81 cuadros | 62 → 77 | 12 → 3 |
| Prueba del cuadrado, 626 cuadros | 216 → 243 | 27 → 18 |

En la segunda grabación el giro máximo entre poses adyacentes bajó de 143,81°
a 15,22°, sin giros mayores de 45°. Esto describe estabilidad de estimaciones;
no equivale a un error físico medido. El informe reproducible está en
`salidas/revision_seguimiento/comparacion.json`. La recuperación sigue exigiendo
evidencia visible: una oclusión completa o desenfoque prolongado ocultan el AR.

## Primera localización con el iPhone, sin cartas

`tablero_iphone.json` usa solo el ArUco ID 0 de 32 mm como referencia fija.
El origen está en su centro; X apunta a su derecha, Y a su borde superior y Z
sale de la pantalla. Si el teléfono está horizontal con la pantalla hacia arriba,
Z apunta hacia arriba. Conservar el zoom de la imagen que se midió y apoyar o
sujetar el teléfono para que no se mueva mientras se desplaza la webcam/notebook.

Desde la carpeta del proyecto:

```bash
.venv/bin/python main.py --solo-localizacion --vista-externa \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --camara 0 --salida salidas/prueba_iphone
```

Se muestran la imagen de la webcam, con ejes y coordenadas, y una segunda ventana
con la escena 3D proyectada desde un observador externo fijo. Esta opción reutiliza
la geometría del visualizador y se actualiza en vivo con OpenCV, sin instalar
Open3D. No permite orbitar la vista con el mouse. El modo Open3D sigue disponible
al omitir `--vista-externa`, una vez instalada su dependencia.

Al ver el ID 0 debe aparecer `CAMARA LOCALIZADA`. Mover lentamente la cámara y
mantener visible el marcador completo con margen blanco; dejar fijo el teléfono.
Si se oculta, se indica pérdida y la última pose queda naranja punteada en la
vista externa. Q o Escape cierran la prueba y guardan imágenes y un reporte en
`salidas/prueba_iphone`.

La primera sesión física ya está guardada: 200 cuadros, de los cuales 100 fueron
anteriores a la primera detección. A partir de ella hubo 79 poses válidas en 100
cuadros, 10 pérdidas breves y 10 recuperaciones. El usuario confirmó teléfono
fijo y notebook/cámara en movimiento. Se observó un salto aislado de 1148,8 mm
y 117,6° entre dos cuadros separados por 0,10 s. El marcador se ve pequeño y
muy escorzado en la última imagen. No se considera todavía una validación de
estabilidad o precisión; el informe de revisión está en
`salidas/prueba_iphone/revision.json` y conserva el reporte original sin cambios.

La segunda sesión (`salidas/prueba_iphone_2`) contiene 81 cuadros y video: 60 poses
válidas (74,1 %), 21 cuadros sin marcador reconocido y 12 recuperaciones. No se
repite el salto grande: máximo cambio adyacente estimado de 10,90 mm y 1,50°.
Estos cambios son movimiento estimado entre cuadros, no una medición de error
físico. Se reprodujeron exactamente sus poses desde las esquinas guardadas.
La continuidad de detección todavía debe mejorarse; el marcador quedó cerca del
borde inferior. El modo de localización ahora dibuja un recuadro de guía: procurar
mantener el marcador completo dentro de él. Informe: `salidas/prueba_iphone_2/revision.json`.

Para una nueva sesión, mantener el mismo marcador de 32 mm y el mismo zoom. Colocar el
teléfono fijo en un soporte, más enfrentado a la webcam y aproximadamente a
25–35 cm si permite enfocar; el cuadrado debe verse bastante más grande que en
la primera sesión. Dejar ambos quietos unos cinco segundos, mover lentamente la
notebook, ocultar brevemente el marcador y volver a mostrarlo. Guardar una nueva
sesión con video para poder reproducir cualquier ambigüedad:

```bash
.venv/bin/python main.py --solo-localizacion --vista-externa --guardar-video \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --camara 0 --salida salidas/prueba_iphone_3
```

Se guarda `camara_cruda.avi` (MJPEG, codificado a 10 fps), además de las imágenes
y el JSON con tiempos reales, esquinas observadas y candidatos PnP por cuadro.
Los tiempos del JSON son la referencia para estudiar velocidades; la tasa del
archivo de video es solo de reproducción. El contador de recuperaciones ya
funciona también con webcam y no cuenta la primera adquisición como recuperación.
El video es opcional y requiere `--salida`; si ya existe uno allí, elegir una
carpeta nueva para conservar ambas sesiones.

Se evaluó un rescate de umbral adaptativo sobre el video comprimido. Pasó de
62 a 67 detecciones, pero introdujo picos de pose de hasta 64,69 mm frente a
11,03 mm del reprocesamiento base. Por eso `rescate_umbral=False` sigue siendo
el valor predeterminado de `Detector`; la aplicación conserva el comportamiento
original. La opción experimental solo debe activarse para análisis con nueva
validación. La comparación MJPEG no sustituye los resultados de la captura cruda.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Las pruebas usan `unittest` de Python. No instalar simultáneamente otras variantes
`opencv-python`/`opencv-python-headless` en el mismo entorno: `opencv-contrib-python`
ya incluye OpenCV y ArUco.

Para la ventana 3D de Open3D con controles de vista, usar un entorno compatible e instalar:

```bash
python -m pip install -r requirements-3d.txt
```

La instalación verificada en esta sesión usa Python 3.13, NumPy 2.5.2 y
opencv-contrib-python 5.0.0.93 (`cv2.__version__ == 5.0.0`). Se recuperaron wheels
de la caché local porque el acceso de red del entorno está restringido. Open3D
no está instalado aquí; su ventana requiere validación en un equipo con pantalla
y un Python compatible, por ejemplo 3.12. El ensayo sin ventanas no lo importa.

## Ensayo reproducible sin cámara

```bash
python -m unittest discover -s pruebas -v
python main.py --simular --sin-ventanas --frames 90 --probar-perdida \
  --verificar-simulacion --salida salidas/demo
```

El segundo comando renderiza imágenes con ArUco, las detecta y ejecuta PnP en cada
cuadro. No entrega las esquinas verdaderas al estimador. Guarda `reporte.json`
con estados y errores, `camara_ar.png` y `vista_externa.png`. La vista externa
exportada permite inspeccionar la geometría sin Open3D y conserva un observador
independiente de la cámara estimada.

Las tolerancias del ensayo raster se fijaron antes de ejecutarlo: error máximo
de cámara **5 mm / 1 grado**, y posición de carta en mundo **12 mm**. El proceso
termina con código 2 si supera esos límites, pierde cuadros fuera del tramo
oculto, no detecta cartas o no recupera seguimiento. Estos límites verifican
el simulador y el pipeline; **no predicen precisión física**. Las pruebas con
esquinas exactas comprueban el álgebra con tolerancias más estrictas.

Para ver la simulación con cámara/AR y escena externa interactiva:

```bash
python main.py --simular
# Si Open3D todavía no está instalado:
python main.py --simular --sin-3d
```

Cerrar con `q` o Escape en la ventana de cámara, o cerrar la ventana 3D. El
simulador mueve la cámara y las cartas de manera independiente. Las pruebas
también aíslan cada movimiento y comprueban pérdida y recuperación del tablero.

## Preparación física pendiente

1. Elegir webcam y resolución, mantener enfoque/zoom estables durante calibración
   y demostración. La aplicación rechaza cuadros de otra resolución.
2. Usar el damero impreso al 100% o mostrarlo completo en una pantalla. Medir el
   lado efectivo de sus cuadrados y tomar 15–25 vistas variadas que cubran el
   campo de visión. En pantalla, conservar la escala medida durante la sesión.
3. Calibrar con esas imágenes y conservar JSON, capturas y RMS de reproyección.
   Un error pequeño por sí solo no prueba buena calibración: revisar cobertura,
   inclinaciones y proyección sobre imágenes no usadas para ajustar.
4. Imprimir los marcadores y medir el **cuadrado negro**, sin incluir margen
   blanco. Medir también las posiciones de sus cuatro esquinas en el tablero.
   El origen propuesto es su centro, con X a la derecha, Y hacia el fondo y Z
   hacia arriba. El diccionario es `4x4_50`.
5. Usar IDs **0–3** para referencias fijas y **10–11** para cartas. Copiar los JSON
   de `ejemplos/`, sustituir las medidas y marcar `es_ejemplo: false` en ambos.
   En cartas, `lado_mm` es el lado del marcador, no el tamaño de la carta.
6. Ejecutar con los archivos medidos:

```bash
python calibrar.py patron --salida salidas/patron.svg
python calibrar.py capturar --camara 0 --ancho 1280 --alto 720 \
  --lado-mm 25 --directorio capturas/webcam --salida calibraciones/webcam.json
python generar_marcadores.py --salida marcadores
python main.py --calibracion calibraciones/webcam.json \
  --tablero tablero_medido.json --cartas cartas_medidas.json --camara 0
```

Reemplazar `25` por el lado medido del casillero, en mm. El patrón predeterminado
de 9×6 esquinas interiores mide 270×205 mm con sus márgenes: imprimir en A4
horizontal al 100%. Los subcomandos de `calibrar.py` permiten generar el patrón,
calibrar imágenes o capturar con webcam; consultar su ayuda. Para repetir el
cálculo a partir de capturas conservadas:

```bash
python calibrar.py imagenes 'capturas/webcam/*.png' --lado-mm 25 \
  --salida calibraciones/webcam.json
```

Las instrucciones exactas de impresión se incluyen también en la ayuda del
generador de marcadores. No imprimir el tablero de 600×400 mm ajustándolo a A4:
cambiaría la escala del mundo.

La primera sesión de calibración real ya está realizada y revisada. Quedan el
montaje con ArUco medidos, la comprobación independiente de precisión y
estabilidad con webcam y la validación de la ventana Open3D.

## Geometría y seguimiento

`T_ab` lleva puntos de `b` a `a`. OpenCV usa cámara X derecha, Y abajo, Z adelante;
el tablero usa Z arriba. PnP entrega `T_cw`, mientras la escena recibe
`T_wc = inversa(T_cw)`. Para una carta, `T_wm = T_wc @ T_cm` solo es válido con
ambas observaciones del mismo cuadro. Las cartas nunca localizan la cámara.

Se trabaja con imágenes **crudas**: detección, PnP y AR usan los mismos `K`,
`dist` y resolución. No se aplica `undistort` antes de volver a aplicar `dist`.
No hay promedio temporal de esquinas en píxeles. La selección de soluciones
planares considera profundidad, orientación física y continuidad entre
soluciones de reproyección similar. Si IPPE no produce candidatos aceptables,
se usa SQPNP refinado con los mismos filtros físicos y de reproyección; esto
resuelve un fallo observado en poses frontales centradas. Un tablero casi
frontal sigue pudiendo ser ambiguo con ruido.

Cuando faltan referencias válidas, `T_cw` y `T_wc` actuales son `None`. La última
pose se conserva aparte con antigüedad y se dibuja diferenciada. Las cartas
pueden seguir mostrando AR respecto de la cámara aunque se pierda el mundo,
pero no reciben una posición mundial inventada.

## Organización y atribuciones

`tp5/modelos.py` define contratos; `configuracion.py` contiene el ejemplo;
`calibracion.py`, `tablero.py`, `deteccion.py`, `pose.py`, `simulador.py`,
`visualizador.py` y `ar.py` implementan los frentes. `main.py` integra y
`pruebas/` verifica geometría y comportamiento. El estado de entrega está en
[PLAN_TP5.md](PLAN_TP5.md).

Base consultada y conservada sin modificaciones:
`../tp3-localizacion-homografica`. Se adapta su detector ArUco, refinamiento de
bordes, generación SVG y modelo pinhole del simulador; las atribuciones de cada
adaptación figuran en los módulos. No se reutiliza su registro homográfico de
cámara quieta para localizar una cámara móvil.

Referencias: [PnP y convenciones de OpenCV](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html),
[Visualizer de Open3D](https://www.open3d.org/docs/release/python_api/open3d.visualization.Visualizer.html),
[calibración de la cátedra](https://github.com/AlejandroSilvestri/VisiAr/tree/main/Calibration).
