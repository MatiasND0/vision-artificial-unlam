# TP5 — Instrucciones compartidas para Codex

## Objetivo

Implementar el Trabajo Práctico 5 de visión artificial: calibrar una cámara,
detectar ArUco, estimar su pose mediante PnP y mostrar su posición y orientación
en un mundo 3D mientras se mueve. La demostración será una mesa de duelo con
cartas de Yu-Gi-Oh y elementos de realidad aumentada.

Leer `PLAN_TP5.md` antes de implementar. Contiene alcance, contratos, responsables,
dependencias y criterios de entrega. Los nombres de módulos son propuestos;
el coordinador puede ajustarlos manteniendo actualizados ambos documentos.

## Proyecto y base

- Trabajar en `/home/hhouse/ws/vis/vision-artificial-unlam` (el usuario trasladó aquí el TP5).
- Usar `/home/hhouse/ws/vis/tp3-localizacion-homografica` como base de consulta.
  Conservar ese proyecto sin modificaciones.
- Reutilizar selectivamente detección, generación de marcadores, anotaciones y
  componentes útiles del simulador; conservar las atribuciones del código.
- La homografía registrada una sola vez en TP3 no reemplaza la localización
  PnP de la cámara en cada cuadro.

## Organización con subagentes

Para implementar los frentes independientes del plan, usar subagentes y reunir
sus resultados en este chat. El agente principal será el coordinador e integrador.

- Mantener como máximo tres subagentes trabajando simultáneamente, además del
  coordinador, respetando el límite disponible en la sesión.
- El coordinador crea primero los contratos y las estructuras compartidas.
- Cada delegación debe indicar objetivo, archivos asignados, entradas y salidas,
  dependencias, verificaciones y qué informe devolver.
- Asignar archivos exclusivos. El coordinador mantiene `main.py`, dependencias,
  modelos compartidos, configuración general y documentación de estado.
- Los subagentes pueden leer todo el proyecto, pero editan únicamente lo asignado.
  Si necesitan cambiar una interfaz compartida, comunican la necesidad al
  coordinador antes de que este integre el cambio.
- Trabajar con datos sintéticos cuando falte un módulo real; no esperar ociosamente
  una dependencia que pueda representarse con el contrato acordado.
- El coordinador integra y verifica los resultados. Finalizar un subagente no
  demuestra que la aplicación completa funcione.
- Para cambios pequeños o tareas dependientes, trabajar secuencialmente.

## Convenciones técnicas esenciales

- Python y OpenCV para captura, ArUco, calibración y PnP; Open3D como visualizador
  inicial. Evitar añadir infraestructura que no ayude a la primera entrega.
- Usar milímetros en toda la geometría del proyecto. Rotular las unidades.
- Mundo: tablero fijo, X a la derecha, Y hacia el fondo, Z hacia arriba.
- Cámara OpenCV: X a la derecha, Y hacia abajo, Z hacia adelante.
- `T_ab` transforma coordenadas del sistema `b` al sistema `a`.
  PnP produce `T_cw`; el visualizador recibe `T_wc = inversa(T_cw)`.
- Separar los marcadores fijos del tablero de los marcadores móviles de cartas.
  Solo las referencias fijas con geometría conocida localizan la cámara en el mundo.
- No aplicar dos veces la corrección de distorsión. Hacer explícita la resolución
  y los parámetros usados por calibración, PnP y proyección AR.
- Una pose sin observación válida debe marcarse como perdida. Una última pose
  conservada para visualizar no cuenta como una medición actual.
- No promediar esquinas en píxeles de cuadros distintos suponiendo cámara quieta.
  Cualquier suavizado para cámara móvil debe respetar el cambio de pose.

## Verificación y comunicación

- Primero comprobar calibración, pose y visualización; después añadir efectos AR.
- Verificar la inversión de transformaciones, el movimiento de cámara contra
  verdad sintética y la independencia entre mover una carta y mover la cámara.
- Diferenciar pruebas automáticas sintéticas de la validación física con webcam.
  No afirmar que una cámara fue calibrada o que una demo real funciona sin evidencia.
- Cada subagente informa archivos cambiados, decisiones, verificaciones realizadas
  y limitaciones pendientes. El coordinador registra avances en `PLAN_TP5.md`.
- Explicar el progreso en español, con mensajes breves y comprensibles.
- Avanzar dentro del alcance acordado sin pedir confirmación por cada archivo o
  etapa. Pedir al usuario los datos físicos que no puedan obtenerse del proyecto,
  como el tamaño medido de un marcador o las capturas de una calibración real.
