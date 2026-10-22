# TP5 — Mesa de duelo AR y localización de cámara con PnP

Estado al 21 de octubre de 2026: tres frentes implementados e integrados con
datos sintéticos. Revisión cruzada realizada, hallazgos corregidos y 88 pruebas
aprobadas. Calibración de webcam real obtenida y revisada con capturas del iPhone.
Referencia ArUco de iPhone configurada en 32 mm. Modo de primera localización sin
cartas y vista externa en vivo con OpenCV probados. Dos sesiones físicas revisadas:
en la segunda no se repitió el salto grande, pero la detección sigue siendo
intermitente. Se integraron los dos modelos GLB, recuperación visual breve,
control de saltos, animación procedural y efectos de invocación. ID 0 y 10
permiten mostrar Dragón y Mago simultáneamente con anclas independientes. Comparaciones
reproducibles muestran menos interrupciones. La precisión física, las cartas
medidas y la ventana Open3D siguen pendientes.

## Cómo vamos a trabajar

El usuario conversa con el coordinador en el chat principal. El coordinador define
interfaces, reparte tareas concretas a subagentes, recibe sus informes, integra los
cambios y comprueba el resultado completo. El usuario puede seguir el progreso sin
tener que dirigir a cada agente por separado.

`AGENTS.md` guarda instrucciones persistentes del proyecto. Este archivo guarda
el plan, los contratos y el estado. Los Markdown no crean subagentes por sí solos:
la delegación la realiza Codex al ejecutar el trabajo. No hace falta un archivo
por agente ni una configuración personalizada para esta primera experiencia.

La preparación incluyó una revisión de solo lectura del reparto. En la sesión
de implementación el coordinador creó primero los contratos y delegó tres frentes
con archivos exclusivos: calibración, PnP/simulación y visualización/AR. Después
integró el pipeline y reasignó al agente de calibración una revisión independiente
de solo lectura, sin exceder tres subagentes simultáneos.

## Resultado de la primera entrega

- Calibrador que obtiene y guarda parámetros intrínsecos y distorsión de la cámara.
- Tablero fijo con marcadores ArUco de tamaño y posiciones conocidos.
- Pose de cámara actualizada con PnP en cada cuadro válido.
- Vista 3D externa con tablero, ejes, representación de la cámara, dirección de
  observación y trayectoria. La vista externa permite apreciar el movimiento.
- Dos cartas identificadas por ArUco en fundas o soportes, con nombre y estadísticas.
- Elementos AR simples anclados a las cartas, inicialmente con geometría o
  ilustraciones sobre planos verticales.
- Simulador con cámara móvil y verdad conocida para verificar la localización.
- Guía de calibración, ejecución y demostración con las limitaciones observadas.

Quedan para otra etapa los duelos completos, reconocimiento por la ilustración de
la carta, oclusión realista por objetos físicos y monstruos 3D animados.

## Lo que aprovechamos de TP3

Base: `/home/hhouse/ws/vis/tp3-localizacion-homografica`.

| Componente existente | Aprovechamiento previsto |
| --- | --- |
| `localizacion/deteccion.py` | Detector ArUco y esquinas refinadas |
| `generar_marcadores.py` | Generación de los marcadores de tablero y cartas |
| `localizacion/anotacion.py` | Estilo de anotaciones y paneles de estado |
| `localizacion/simulador.py` | Proyección sintética, texturas y modelo de cámara |
| `diagnostico_camara.py` | Diagnóstico, adaptando el acceso a cámara al sistema |

El registro homográfico y el historial usado para registrar una cámara quieta no
se trasladarán directamente a una cámara móvil. El simulador actual mantiene la
cámara fija; habrá que exponer sus parámetros y añadir movimiento de cámara.
La captura actual de `main.py` fuerza un backend de Windows: en este entorno Linux
el coordinador debe seleccionar un backend compatible.

## Equipo y propiedad de archivos propuesta

Los módulos de esta tabla ya están implementados. El coordinador agrega además
`requirements-3d.txt`, `ejemplos/` y `pruebas/test_integracion.py`.

| Responsable | Trabajo | Archivos exclusivos |
| --- | --- | --- |
| Coordinador | Base, contratos, captura, integración y documentación | `main.py`, `tp5/modelos.py`, `tp5/configuracion.py`, `tp5/fuentes.py`, `tp5/__init__.py`, `requirements.txt`, `.gitignore`, `README.md`, `AGENTS.md`, `PLAN_TP5.md` |
| Agente de calibración | Captura del patrón, calibración, persistencia y validación | `calibrar.py`, `tp5/calibracion.py`, `pruebas/test_calibracion.py` |
| Agente de localización | Tablero, estimación PnP, transformaciones y simulador | `tp5/deteccion.py`, `tp5/tablero.py`, `tp5/pose.py`, `tp5/simulador.py`, `generar_marcadores.py`, `pruebas/test_pose.py` |
| Agente de visualización | Escena externa 3D y proyección de elementos AR | `tp5/visualizador.py`, `tp5/ar.py`, `assets/`, `pruebas/test_proyeccion_ar.py` |

Los nombres son roles de trabajo, no agentes personalizados que haya que instalar.
Los archivos comunes tienen un único editor. Si los agentes comparten el mismo
directorio, no se aíslan automáticamente sus cambios: se respeta la propiedad de
archivos. Un revisor puede trabajar después de liberar uno de los tres puestos.

## Contratos que el coordinador debe fijar antes de delegar

1. **Calibración:** matriz `K` de 3×3, coeficientes `dist`, tamaño de imagen
   `(ancho, alto)`, error de reproyección en píxeles y metadatos del patrón y sus
   medidas. El formato de archivo y su versión se acuerdan al crear los modelos.
2. **Detección:** identificador y cuatro esquinas en píxeles, conservando el orden
   canónico del detector ArUco, asociado al mismo cuadro de captura.
3. **Tablero:** diccionario ArUco y, por cada marcador fijo, sus cuatro esquinas
   3D en el mundo, expresadas en milímetros y en correspondencia con la detección.
   Los marcadores de cartas se configuran por separado, con sus medidas reales.
4. **Estado por cuadro:** `frame_id`, tiempo, validez del seguimiento, marcadores
   usados, error de reproyección y transformaciones actuales. Una pose inválida
   lleva transformaciones actuales ausentes; la última pose válida, si se
   conserva, va en un campo separado con su antigüedad.
5. **Visualizador:** recibe un estado del mundo y actualiza geometría. No detecta
   marcadores ni ejecuta PnP. Mantiene su observador externo independiente de la
   pose estimada, para que el movimiento de la cámara sea visible.
6. **AR:** recibe imagen, calibración y pose de cada carta respecto de la cámara.
   La composición de una carta en el mundo requiere además una pose de cámara
   válida del mismo cuadro.

Convenciones geométricas:

- Unidades: milímetros. Mundo con X derecha, Y fondo y Z arriba; origen fijo en
  un punto definido del tablero.
- Cámara: convención de OpenCV, X derecha, Y abajo y Z adelante.
- `T_ab` lleva puntos de `b` a `a`. PnP con puntos del tablero entrega `T_cw`.
  La pose de cámara en el mundo es `T_wc = inversa(T_cw)`.
- Si una carta tiene transformación `T_cm`, entonces `T_wm = T_wc @ T_cm`.
- No usar IDs móviles de cartas como referencias fijas para localizar la cámara.
- Para un marcador cuadrado individual, respetar el orden de esquinas requerido
  por el método elegido. `IPPE_SQUARE` se reserva al cuadrado individual; no se
  aplica directamente a todas las esquinas de un tablero con varios marcadores.
- Con geometría plana, evaluar las soluciones físicamente válidas y la continuidad
  temporal; no asumir que un error de reproyección pequeño elimina toda ambigüedad.

## Secuencia de ejecución

### Interfaces concretas de la implementación

Los modelos están en `tp5/modelos.py`; la configuración de ejemplo está en
`tp5/configuracion.py`. Los agentes importan esos contratos sin modificarlos.

- Calibración: `calibrar_desde_puntos(puntos_objeto, puntos_imagen, tamano,
  metadatos=None) -> Calibracion`; `guardar_calibracion(calibracion, ruta)`;
  `cargar_calibracion(ruta) -> Calibracion`. JSON versión 1 con `unidades: "mm"`,
  `K`, `dist`, `tamano`, `rms_px`, `metadatos`. Imágenes crudas, sin undistort.
- Tablero: `crear_tablero_ejemplo() -> Tablero`; `guardar_tablero(tablero,ruta)`;
  `cargar_tablero(ruta) -> Tablero`. Marcadores 0–3 de 60 mm en centros configurados.
  Cartas móviles 10–11 de 40 mm, excluidas del tablero.
- Detección: `Detector(diccionario="4x4_50").detectar(imagen) -> list[Deteccion]`.
- Pose: `invertir_transformacion(T)`, `transformar_puntos(T,puntos)`;
  `EstimadorPose(tablero, calibracion, error_max_px=3.0).estimar(detecciones,
  frame_id, tiempo_s) -> PoseCamara`; `estimar_cartas(detecciones, calibracion,
  cartas_config, pose_camara) -> list[PoseCarta]`. Esquinas locales de cartas:
  `[-L/2,+L/2,0], [+L/2,+L/2,0], [+L/2,-L/2,0], [-L/2,-L/2,0]`.
- Simulación: `Simulador(tablero=None, tamano=(1280,720), semilla=0)` expone
  `.calibracion`; `.generar(frame_id, mover_camara=True, mover_cartas=True,
  ocultar_tablero=False) -> CuadroSintetico`. Contiene imagen renderizada y
  esquinas exactas, estas últimas solo para pruebas geométricas.
- AR: `dibujar_ar(imagen, calibracion, estado, cartas_config) -> imagen`;
  proyección con K/dist sobre imagen cruda, sin doble corrección.
- Visualización: `Visualizador3D(tablero, calibracion, habilitado=True)`;
  `.actualizar(estado) -> bool` (False al cerrar), `.cerrar()`. Importar Open3D
  de forma diferida; modo deshabilitado verificable sin pantalla ni Open3D.
- Primera prueba física: `--solo-localizacion` omite las cartas y no exige su
  archivo; `--vista-externa` actualiza una segunda ventana OpenCV con la escena
  3D proyectada desde un observador fijo, sin importar Open3D. Ambos modos
  conservan los requisitos de calibración real y tablero medido para webcam.
- Observabilidad: `EstadoMundo.detecciones` contiene las esquinas observadas en
  el mismo cuadro; `EstimadorPose.diagnostico` conserva métodos, candidatos,
  errores e índice/criterio de selección, reemplazándose al iniciar cada cuadro.
  `--salida` los guarda en el JSON; `--guardar-video` conserva además imagen
  cruda MJPEG para revisión. Se mantienen los métodos y umbrales existentes.
- El coordinador mantiene captura, CLI, pruebas de integración y documentación.
  Python 3.13 disponible; Open3D puede requerir un entorno Python 3.11/3.12.

### Etapa 0 — Preparación

- [x] Acordar la idea de una mesa de duelo AR.
- [x] Identificar la base del TP3 y la ruta de destino `/home/hhouse/ws/vis/tp5`.
- [x] Preparar instrucciones, reparto, contratos y criterios de entrega.
- [x] Registrar el resultado de la revisión del subagente.

Revisión realizada: se confirmó la necesidad de fijar interfaces antes de
paralelizar, invertir la transformación PnP para visualizar la cámara y separar
referencias fijas de cartas móviles. Se incorporaron los riesgos del promedio de
esquinas, el simulador con cámara fija y la ambigüedad planar. Se mantienen
milímetros como unidad común por compatibilidad con TP3; si una herramienta externa
requiere otra unidad, la conversión será explícita en su adaptador.

### Etapa 1 — Base común, a cargo del coordinador

- [x] Crear la estructura mínima y trasladar selectivamente la base necesaria.
- [x] Definir modelos compartidos y ejemplos sintéticos de sus entradas y salidas.
- [x] Establecer diccionario, distribución inicial de IDs y formato del tablero.
- [x] Elegir formato de calibración y validar resolución y unidades.

Esta etapa precede al trabajo de implementación paralelo para evitar interfaces
incompatibles. No requiere una calibración real para producir datos sintéticos.

### Etapa 2 — Tres frentes en paralelo

- [x] Calibración: implementar su herramienta y probar guardado, carga y validación.
- [x] Localización: implementar tablero, PnP y cámara móvil simulada.
- [x] Visualización: construir vista externa y AR con poses sintéticas del contrato.

El agente de localización puede empezar con intrínsecos conocidos del simulador.
El de visualización puede empezar con estados artificiales. La medición real de la
cámara y el tablero será necesaria para la integración física.

### Etapa 3 — Integración y verificación

- [x] Integrar captura → detección → PnP → pose de cámara → vista 3D (geometría y PNG verificados; ventana pendiente).
- [x] Integrar detección de cartas y proyección AR sin confundir sus referencias.
- [x] Revisar pérdida y recuperación de seguimiento, escalas y orientación de ejes.
- [x] Ejecutar una revisión independiente y corregir los hallazgos relevantes.
- [x] Realizar y revisar la calibración física con capturas de webcam.
- [x] Documentar la primera prueba física de localización con ArUco medidos.
- [ ] Repetir con mejor visibilidad y comprobar estabilidad sin saltos aislados.

### Etapa 4 — Presentación del TP

- [x] Documentar instrucciones reproducibles y teoría necesaria para defenderlo.
- [x] Preparar una demostración sintética breve y guardar resultados verificables.

Dentro del alcance de implementación, avanzar entre etapas sin pedir aprobación
por cada módulo. Las tareas que requieren mover la cámara, imprimir patrones o
medir objetos se coordinan con el usuario cuando llegue ese momento.

## Criterios de aceptación

| Comprobación | Evidencia esperada |
| --- | --- |
| Cámara calibrada | Archivo con parámetros, resolución, medidas y error documentado de capturas reales |
| Pose correcta en simulación | Comparación de posición y orientación estimadas contra verdad conocida en distintas poses; tolerancias explícitas antes de evaluar |
| Transformaciones coherentes | Composición de una transformación y su inversa produce identidad; proyección consistente con las esquinas observadas |
| Cámara móvil | La representación 3D cambia de posición y orientación y el tablero conserva su referencia |
| Carta móvil | Al mover solo una carta, cambia su ubicación y el objeto AR la acompaña; la cámara permanece estable dentro del error medido |
| Pérdida de referencia | Se indica seguimiento perdido y no se presenta la última pose como una observación nueva |
| Recuperación | Al reaparecer referencias válidas, se recupera la localización sin redefinir el mundo |
| AR anclada | Con tablero y cartas quietos, los elementos conservan su anclaje mientras se mueve la cámara |

Primero usar pruebas geométricas sintéticas que puedan detectar errores de ejes,
escala e inversión. La prueba del visualizador requiere inspección visual. Las
pruebas sintéticas no sustituyen la validación con la cámara real.

## Intervenciones físicas pendientes

- Webcam 0 a 1280×720 seleccionada y calibrada; mantener estable el enfoque y zoom.
- Patrón mostrado en iPhone: 10 casillas medidas en 74 mm, lado de casilla 7,4 mm.
- 25 imágenes capturadas y revisadas. Añadir cobertura superior y derecha antes de
  la validación final de precisión sobre todo el campo de visión.
- Mostrar o montar referencias ArUco fijas y medir tamaños y posiciones.
- Preparar dos cartas con ArUco visibles y margen blanco suficiente.

La calibración ya usa medidas reales del patrón. Las dimensiones del tablero
ArUco y de las cartas siguen siendo ejemplos hasta medir sus montajes.

## Revisión de la calibración física del iPhone

Archivo: `calibraciones/webcam.json`; capturas: `capturas/iphone/vista_000.png`
a `vista_024.png`. Webcam 0 a 1280×720, patrón de 9×6 esquinas interiores y
casilla de 7,4 mm según los 74 mm medidos por el usuario sobre 10 casillas.

- Se redetectó el patrón en las 25 imágenes, todas de la resolución esperada.
- RMS guardado: 0,2782815 px; RMS recalculado: 0,2782815 px.
- Máximo RMS por captura: 0,4478015 px.
- Validación cruzada: cinco ajustes con 20 vistas, reservando otras cinco en cada
  partición; máximo RMS de una vista reservada: 0,4534790 px.
- Rango de focal horizontal entre ajustes: 842,09–854,85 px; vertical:
  836,53–848,27 px. No se sustituyeron los parámetros originales guardados.
- Cobertura de esquinas observadas: X 126,5–973,2 px, Y 271,1–666,0 px. Hay poca
  información directa de la parte superior y del extremo derecho del campo.

Informe: `salidas/revision_calibracion_iphone.json`. La inspección visual de las
capturas y sus detecciones permite avanzar a una primera prueba de PnP. Las vistas
reservadas pertenecen a la misma sesión: esto no certifica exactitud en milímetros
ni reemplaza una comprobación independiente con ArUco de tamaño medido.

Se preparó `salidas/aruco_iphone_id0.png` (diccionario 4x4_50), detectado y
verificado como ID 0. El usuario midió **32 mm de lado negro** y se creó
`tablero_iphone.json`: único marcador fijo, esquinas a ±16 mm y origen en su
centro. Conservar su escala de visualización y mantener fijo el teléfono.

Se añadió el modo `--solo-localizacion` para iniciar la prueba sin inventar
medidas de cartas todavía inexistentes. `--vista-externa` muestra en vivo la
misma geometría 3D mediante el renderizador de diagnóstico OpenCV; no requiere
Open3D. Una regresión verifica la recuperación de la escala de un marcador
sintético de 32 mm por el camino de captura, y otra comprueba que la vista se
actualiza por cuadro sin importar Open3D. Total actual: **53 pruebas aprobadas**.
Se abrieron las dos ventanas con 30 cuadros simulados: 30 poses válidas y salida
correcta. También sigue pasando el ensayo completo de 90 cuadros con cartas,
pérdida y recuperación. La primera prueba física se documenta a continuación.

## Primera prueba física de localización

Evidencia original: `salidas/prueba_iphone/reporte.json`, `camara_ar.png` y
`vista_externa.png`. Revisión separada: `salidas/prueba_iphone/revision.json`.
El usuario confirmó que movió la notebook/cámara y mantuvo fijo el iPhone.

- 200 cuadros a aproximadamente 9,5 cuadros/s; los primeros 100 no tenían una
  referencia fija detectada y se informan separados de la fase de seguimiento.
- Desde la primera detección: 79 poses válidas en 100 cuadros, 10 pérdidas y
  10 recuperaciones. Todas las poses válidas usaron únicamente ID 0.
- Reproyección de poses válidas: mediana 0,0667 px, percentil 95 de 0,2214 px y
  máximo 0,3528 px. No representan un error físico medido en milímetros.
- Se observó un valor aislado en el cuadro 148: entre 148 y 149, el cambio
  estimado fue 1148,77 mm y 117,59° en 0,0999 s. No se considera estabilidad
  suficiente para la validación final.
- La imagen final muestra el marcador pequeño y escorzado. Una revisión con
  subagente encontró que esa geometría admite una solución alternativa de bajo
  residual, compatible con ambigüedad planar. Sin las esquinas ni el video de
  ese cuadro no puede demostrarse la causa concreta del salto.

Se corrigió un error del resumen que solo contaba recuperaciones al simular.
Ahora también cuenta las de webcam, separando adquisición inicial y seguimiento.
El reporte original no se reescribió; las diez recuperaciones se calcularon
directamente de su secuencia de estados y quedaron en el informe de revisión.

Para obtener evidencia reproducible se añadió diagnóstico PnP por cuadro y video
crudo opcional. El subagente editó únicamente `tp5/pose.py` y su regresión
geométrica; el coordinador integró observaciones, video y métricas en el bucle.
No se ensancharon umbrales ni se aplicó suavizado a las poses. La regresión de
integración reproduce adquisición, pérdida y recuperación con captura simulada,
verifica el contador y redetecta el marcador en los cuatro cuadros del video
guardado. La siguiente sesión debe usar el mismo marcador de 32 mm más cerca y
menos escorzado, con el teléfono fijo; conservarla en `salidas/prueba_iphone_2`.

## Segunda prueba física y evaluación de detección

Evidencia: `salidas/prueba_iphone_2/reporte.json`, `camara_cruda.avi`,
`camara_ar.png` y `vista_externa.png`. Informe: `revision.json` en esa carpeta.

- 81 cuadros en 7,92 s entre la primera y última observación, 60 poses válidas
  (74,1 %), 21 pérdidas por ausencia de detección ArUco y 12 recuperaciones.
- No reaparece el salto de 1,15 m. Mayor cambio entre poses válidas consecutivas:
  10,90 mm; máximo cambio angular adyacente: 1,50°. No es un error físico medido.
- RMS de reproyección mediano: 0,1156 px; percentil 95: 0,2702 px; máximo:
  0,4880 px. Todas las poses usan ID 0.
- Las observaciones originales guardadas reproducen exactamente las 60 poses,
  sin discrepancias de validez. Las dos ramas quedan separadas por al menos
  1,596 px de RMS; no se observó alternancia entre ramas en esta sesión.
- Lados medianos del marcador: 73–79 px. Está cerca del borde inferior, con
  margen mínimo de 5,92 px. Añadir padding o relajar solo el filtro de proximidad
  al borde no recuperó las pérdidas al reprocesar el video; no se atribuye toda
  la intermitencia únicamente al encuadre.

Un subagente evaluó un barrido acotado de ventanas de umbral sin editar PnP,
decodificación ni corrección de bits. En MJPEG, el detector base encuentra 62/81
cuadros y un segundo intento 5–35/10 solo si el primero queda vacío encuentra
67/81, conservando exactamente las detecciones base y sin IDs inesperados.
Sin embargo, tres rescates introducen picos: el mayor cambio adyacente sube de
11,03 a 64,69 mm. No se habilitó como mejora de producción.

`Detector(..., rescate_umbral=False)` conserva la base de forma predeterminada.
El rescate queda como opción experimental explícita para diagnóstico. Tres
regresiones sintéticas prueban recuperación de un marcador degradado con geometría
conocida, conservación exacta de detecciones/refinamiento y negativos sin IDs;
no contienen capturas personales. El coordinador reprodujo el video con ambos
modos y guardó la comparación en `salidas/prueba_iphone_2/reprocesado/reporte.json`.
Las cifras MJPEG no son equivalentes a los cuadros originales sin compresión.

Se añadió una guía de encuadre al modo de localización; no filtra ni modifica
las mediciones. La siguiente captura, si se realiza, debe mantener el marcador
completo dentro de esa guía y guardarse en una carpeta nueva, por ejemplo
`salidas/prueba_iphone_3`. No requiere repetir calibración ni medir de nuevo si
se mantienen cámara, resolución, enfoque y tamaño del marcador.

## Primer objeto AR: cuadrado mundial

Se añadió `--cuadrado-ar`, compatible con `--solo-localizacion` para usar la
calibración real y `tablero_iphone.json` sin cartas ni modelos externos. El
cuadrado mide 40×40 mm, está centrado en (0,0,20) mm y es paralelo al plano del
marcador. Su altura +Z sale 20 mm de la pantalla del iPhone hacia su cara visible.

El subagente modificó exclusivamente `tp5/ar.py` y sus pruebas geométricas;
el coordinador integró CLI, reporte, regresión del pipeline y documentación.
`geometria_cuadrado(lado_mm=40.0, altura_mm=20.0)` devuelve los cuatro vértices
mundiales. `dibujar_ar(..., cuadrado_ar=False)` conserva el modo anterior;
al activarlo usa la `T_cw` actual y K/dist sobre la imagen cruda. El relleno
translúcido se omite si falta pose o un vértice cruza el plano de cámara. No usa
la última pose ni modifica PnP, detección o calibración.

Verificación: 53 pruebas aprobadas, incluidas proyección con cámara móvil,
distorsión, pérdida/recuperación y captura simulada con la nueva opción. Además
se redetectaron los 81 cuadros del video MJPEG de la segunda prueba física:
62 poses válidas con cuadrado visible y 19 sin pose con el cuadrado oculto.
La redetección de MJPEG puede diferir de la captura cruda original (60 válidas).
La comparación del teléfono con/sin AR se inspeccionó visualmente. Evidencia en
`salidas/cuadrado_ar/comparacion_cuadrado.png` y `revision_cuadrado.json`.
Es una reproyección de una grabación anterior, no una nueva medición de precisión
física. El cuadrado aparecerá intermitentemente mientras la detección lo sea.

```bash
.venv/bin/python main.py --solo-localizacion --cuadrado-ar --vista-externa \
  --calibracion calibraciones/webcam.json --tablero tablero_iphone.json \
  --camara 0 --salida salidas/prueba_cuadrado
```

## Continuidad visual y animación de la demo GLB

El proyecto fue trasladado a `/home/hhouse/ws/vis/vision-artificial-unlam`.
Se actualizó AGENTS y se enlazó el entorno existente como `.venv`; TP3 sigue
sin modificaciones. Los dos GLB con créditos ya estaban integrados y funcionaban.

Para el problema informado de apariciones/desapariciones se creó
`DetectorContinuo` en `tp5/seguimiento.py`. Conserva la detección normal y añade
ROI ampliada sobre el cuadro actual y, hasta 0,3 s desde una lectura ArUco,
flujo LK con ida/vuelta, homografía robusta y bits actuales del mismo ID. No
retiene una pose perdida ni promedia esquinas temporales. Solo rescata las
referencias fijas configuradas. La webcam lo activa por defecto; el modo base
y el ensayo sintético permanecen disponibles. Se registran fuentes por detección.

El control opcional de continuidad PnP, habilitado en la webcam, filtra ramas
que exceden cambios compatibles durante intervalos ≤0,5 s. No amplía el umbral
de reproyección ni elimina los candidatos del diagnóstico. Si ninguna rama
cumple, declara pérdida; después de un intervalo largo permite relocalizar.

Se reprocesaron los mismos videos MJPEG de Descargas, conservando las marcas de
tiempo de los reportes: 62→77 poses/81 y 12→3 apariciones separadas en la segunda
prueba; 216→243 poses/626 y 27→18 apariciones en la prueba del cuadrado. En esta
última, el mayor giro adyacente bajó de 143,81° a 15,22° y desaparecieron los
giros mayores de 45°. Son métricas de seguimiento, no exactitud física ni una
nueva prueba de webcam. Informe: `salidas/revision_seguimiento/comparacion.json`.

El subagente modificó exclusivamente `tp5/render_3d.py`, `tp5/animacion.py` y sus
pruebas; el coordinador mantuvo detección, PnP, integración y documentación.
El dragón no trae skin ni clips: el aleteo de sus membranas superiores es
procedural y los brazos conservan la pose base. El mago usa respiración suave
en lugar de reproducir su clip glTF. La demo activa animación y efectos por
defecto: anillos y partículas métricos, distorsionados con K/dist una sola vez,
dibujados detrás de la malla. Todo exige una pose actual. Los originales GLB
se conservan intactos y el tiempo relativo de cuadros permite reproducir el efecto.

## Dos personajes con ArUco independientes

`--por-marcador` configura ID 0 → modelo 0 (Dragón Blanco) e ID 10 → modelo 1
(Mago Oscuro). Conserva `--demo-yugioh` y sus teclas como modo anterior.
El ID 0 sigue siendo referencia fija; el ID 10 se estima como marcador móvil
mediante PnP individual y no interviene en la localización mundial de cámara.
Cuando falta ID 0, el Mago puede seguir proyectándose con T_cm actual; T_wm y
las transformaciones mundiales de cámara permanecen ausentes. La vista indica
que falta la referencia mundial. Cada personaje desaparece solo si pierde su
ancla actual. El rescate visual de ID 10 no lo transforma en referencia fija.

`InstanciaModelo` y `EstadoMundo.instancias_3d` guardan ID, índice de modelo,
cuadro, T_cm y T_wm opcional. `tp5/anclas.py` construye marcos locales de
referencias fijas y poses móviles del mismo cuadro. Se guardan en el reporte.
El subagente extendió el render y sus pruebas exclusivamente; el coordinador
integró anclas, CLI, modelos compartidos, seguimiento, imágenes y documentación.
`dibujar_instancias` usa un único framebuffer con profundidad compartida,
transformación propia por personaje, animación y efectos por ancla. Los recursos
GPU se preparan una sola vez. La selección del modo anterior no se altera.

Se generaron PNG individuales ID 0 e ID 10 en `assets/marcadores`, con el mismo
lienzo 2400×1800 y cuadrado negro 1200×1200 que el ID 0 medido por el usuario.
El tamaño predeterminado del ID 10 es 32 mm, pendiente de comprobación en su
soporte; `--lado-mago-mm` permite indicar otro tamaño. No se declara una nueva
medición física: el reporte distingue valor predeterminado y opción explícita.
La lámina doble requiere medir ambos cuadrados y actualizar la geometría fija
si cambia su escala. Las imágenes individuales permiten alternar en el iPhone
con la misma escala; con ambos visibles, se muestran los dos personajes.

Verificación sintética completa de 90 cuadros: 81 poses de cámara válidas,
9 pérdidas previstas, 1 recuperación y 0 cartas faltantes. El Mago se conserva
relativo a cámara durante las pérdidas mundiales. Error máximo sintético de
carta 1,96157 mm, dentro del límite previo de 12 mm. Total: 88 pruebas, incluidas
independencia al mover el Mago, ocultación individual y oclusión GPU compartida
invariante al orden de entrada. No sustituye una prueba física con el nuevo ID.

## Implementación y evidencia de esta entrega

Se crearon los contratos antes de delegar. Los tres agentes trabajaron en sus
archivos exclusivos; el coordinador integró captura, CLI, dependencias, ejemplos,
documentación y pruebas de integración. TP3 conserva su árbol de trabajo limpio
(`git status --porcelain` sin salida); no se modificó el proyecto base.

Entorno verificado: Python 3.13.5, NumPy 2.5.2, opencv-contrib-python 5.0.0.93
(`cv2.__version__` 5.0.0). Dependencias recuperadas desde caché local debido a
restricciones de red. Open3D se instala aparte mediante `requirements-3d.txt`.
Durante esa integración sintética no se abrió una webcam. La sesión física
posterior con el usuario y su revisión se documentan en la sección anterior.

Verificaciones realizadas:

```bash
.venv/bin/python -m unittest discover -s pruebas -v
.venv/bin/python main.py --simular --sin-ventanas --frames 90 \
  --probar-perdida --verificar-simulacion --salida salidas/demo
.venv/bin/python generar_marcadores.py --salida marcadores
.venv/bin/python calibrar.py patron --salida salidas/patron_calibracion.svg
```

Las **88 pruebas** actuales se reparten en calibración (10), pose/simulación (17),
visualización/AR (17), integración (11), detección (3), seguimiento (6), animación (5),
anclas independientes (4) y render GPU (15). Cubren JSON y unidades, recuperación de
intrínsecos/distorsión, resolución, transformaciones inversas, trayectoria con
verdad conocida, cámara y cartas independientes, distorsión sobre imágenes
crudas, rechazo de geometría físicamente inválida, pérdida y recuperación,
profundidad de AR y descarte de cartas de cuadros anteriores.

El ensayo integrado usa **imágenes rasterizadas → detector ArUco → PnP**;
el estimador no recibe las esquinas exactas. Tolerancias fijadas previamente
en `main.LIMITES_SINTETICOS` y en las pruebas:

| Magnitud | Límite sintético | Máximo observado, 90 cuadros |
| --- | --- | --- |
| Posición de cámara | 5 mm | 0,191617 mm |
| Orientación de cámara | 1 grado | 0,0100964 grados |
| Posición de cartas en mundo | 12 mm | 2,205435 mm |

Resultado: 81 poses válidas, 9 pérdidas previstas al ocultar las referencias,
0 falsas poses válidas, 1 recuperación y 0 cuadros con cartas incompletas.
`verificacion_ok: true`; código de salida 0. Son errores sintéticos, no medidas
de una webcam ni estimaciones de su precisión futura.

Evidencias generadas: `salidas/demo/reporte.json`, `camara_ar.png` y
`vista_externa.png`. Las dos imágenes se inspeccionaron: nombres/ATK/DEF,
geometría vertical y ejes visibles; observador externo fijo con cámara elevada,
frustum, trayectoria y tablero. Se guardaron e inspeccionaron además
`camara_perdida.png` y `vista_perdida.png`, con última pose naranja punteada.
La imagen externa es una proyección diagnóstica
OpenCV de la misma geometría; **no es una captura de una ventana Open3D**.

La revisión independiente encontró y permitió corregir:

1. IPPE/IPPE_SQUARE podía perder soluciones exactas frontales y centradas en
   OpenCV 5. Se agregó rescate SQPNP refinado solo cuando no quedan candidatos
   aceptables, manteniendo profundidad, orientación y umbral de reproyección.
   Hay regresiones a 0 y aproximadamente 3 grados, además de rechazo del dorso
   de carta y de cámara bajo el tablero.
2. El verificador aceptaba cartas ausentes en cuadros intermedios. Ahora exige
   las cartas esperadas en cada cuadro y registra faltantes/inesperadas. Una
   regresión que elimina cartas en 29 de 30 cuadros termina con código 2.
3. Se corrigió la guía física para marcar `es_ejemplo: false` tanto en el tablero
   como en las cartas después de sustituir las medidas.

Límites pendientes: montaje físico y prueba con ArUco medidos, comprobación
independiente de precisión y ejecución e inspección de la ventana Open3D. La
geometría de la escena, sus actualizaciones y la proyección AR se verificaron
sin esa dependencia. La API de detección no lleva un `frame_id` por marcador:
el pipeline debe entregar detecciones y pose del mismo cuadro; `main.py` lo
hace en una única llamada y no conserva detecciones anteriores.

## Fuentes de consulta

- [Subagentes en Codex](https://learn.chatgpt.com/docs/agent-configuration/subagents).
- [Instrucciones con AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
- [Calibración propuesta por la cátedra](https://github.com/AlejandroSilvestri/VisiAr/tree/main/Calibration).
- [PnP y convenciones de OpenCV](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
