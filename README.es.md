**[English](README.md) | [Español](README.es.md)**

# Flotation Plant Optimization: What the Data Supports

[![CI](https://github.com/Rxyxs/flotation-plant-silica-soft-sensor/actions/workflows/ci.yml/badge.svg)](https://github.com/Rxyxs/flotation-plant-silica-soft-sensor/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue) ![Datos](https://img.shields.io/badge/datos-reales%20(Kaggle%2C%20CC0)-2ea44f) ![Licencia](https://img.shields.io/badge/licencia-MIT-green)

Con seis meses de datos reales de una planta de flotación de mineral de hierro, el mismo modelo de gradient boosting que logra un R² de 0,83 con filas de 20 segundos mezcladas al azar es peor que predecir el promedio cuando se valida en orden temporal; solo un modelo que recibe los últimos resultados de laboratorio le gana a repetir el último, y cuatro modelos entrenados con meses distintos recomiendan mover el pH y la densidad de pulpa en direcciones opuestas para las mismas horas, así que los datos no permiten recetar setpoints.

## Lo que encontré

| Hallazgo | Evidencia |
|---|---|
| **Un R² alto acá viene de la validación, no del modelo** | El mismo XGBoost logra R² 0,83 con filas de 20 segundos al azar usando el hierro del concentrado como entrada (sale de la misma muestra de laboratorio que la sílice), 0,46 sin él, 0,28 con horas al azar y -0,43 con horas en orden temporal: peor que predecir el promedio. |
| **Un sensor virtual puro no funciona en esta planta** | Solo con variables de proceso, Ridge mejora al promedio del entrenamiento en 4% del error cuadrático (Diebold-Mariano p = 0,30, sin ganancia detectable) y XGBoost y CatBoost quedan 26-27% peor que él. |
| **La señal la traen los resultados recientes de laboratorio** | Sumando el último resultado de laboratorio conocido dos horas antes, Ridge llega a un RMSE de 0,842 puntos de sílice contra 0,932 de la persistencia (repetir ese último resultado), 10% menos (p < 0,001) y 37% bajo el error del promedio. Los modelos de árboles ganan menos y XGBoost no le gana a la persistencia. |
| **Cuatro modelos, cuatro respuestas** | Para las 40 horas con más sílice de las últimas cinco semanas, cada uno de cuatro modelos entrenados con tramos distintos de la historia promete bajar la sílice entre 0,28 y 0,43 puntos moviendo los cuatro controles a lo más 10%; coinciden en la dirección del movimiento solo en el 10% (densidad de pulpa) a 52% (amina) de las horas, y para el pH el modelo más antiguo dice bajar y los otros tres subir. |
| **Que dos optimizadores coincidan no prueba nada** | Un algoritmo genético y la evolución diferencial encuentran el mismo óptimo sobre el mismo modelo (diferencia mediana de 0,00 puntos). La primera versión de este proyecto usaba esa coincidencia como validación; solo muestra que los optimizadores funcionan, no que el modelo tenga razón. |
| **La razón probable: los operadores reaccionan a la sílice** | La dosis de amina se correlaciona más con la sílice medida dos horas *antes* (0,23) que con la de la misma hora (0,15). Un modelo lee entonces una dosis alta como señal de sílice alta y recomienda menos amina, lo contrario de lo que hace la amina en flotación inversa, donde es el colector que hace flotar la sílice. |

## Los datos

[Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) (Kaggle, `edumagalhaes`, licencia CC0-1.0, descargado el 2026-10-08): 737.453 filas de una planta de flotación catiónica inversa de **mineral de hierro**, del 10 de marzo al 9 de septiembre de 2017. Las variables de proceso (flujo de almidón y de amina, flujo, pH y densidad de la pulpa, flujo de aire y nivel de siete columnas) se registran cada 20 segundos; el hierro y la sílice de la alimentación y del concentrado vienen del laboratorio. El objetivo es la sílice del concentrado, la impureza que la planta quiere baja.

No es cobre ni es Chile: no existen datos públicos de flotación de una planta chilena de cobre, y este es el dataset real de flotación abierto más completo. Las preguntas (qué se puede predecir con los sensores de la propia planta, y si un modelo puede elegir las dosis de reactivo) son las mismas para una planta de cobre.

Lo que necesitó el archivo antes de cualquier modelo, todo resuelto en `src/data.py` y con tests:

- **Las marcas de tiempo solo traen la hora.** Las 180 lecturas de cada hora comparten la misma marca, así que la unidad de análisis es la hora: 4.097 horas, con medias y desviaciones estándar de las lecturas de 20 segundos.
- **Parte de la sílice está interpolada.** En 310 horas cambia dentro de la hora y 232 horas caen sobre rectas perfectas entre dos valores; en total, 328 horas no son resultados de laboratorio y nunca se usan como objetivo.
- **El ensayo de alimentación no es horario.** Cambia solo en el 7,5% de las horas y una vez se mantiene igual 792 horas (33 días), así que entra como "último ensayo" más la antigüedad de ese ensayo.
- **Un hueco de 13 días** (319 horas) en marzo de 2017. Los rezagos nunca lo cruzan.
- **El hierro del concentrado nunca es una entrada**: es la misma muestra de laboratorio que la sílice (correlación -0,80) y no se conoce antes que ella.

![Sílice horaria del concentrado](outputs/figures/silica_series.png)

El objetivo, hora por hora: en azul los resultados de laboratorio, en rojo las horas interpoladas (excluidas). La sílice promedia 2,3% con una desviación estándar de 1,1 puntos y una autocorrelación a una hora de 0,77, que es lo que hace tan favorables las particiones al azar.

## 1. De dónde sale un R² alto

![R² con cuatro protocolos de validación](outputs/figures/validation_protocols.png)

El mismo modelo XGBoost con cuatro protocolos. Las filas de 20 segundos al azar ponen lecturas de la misma hora, con el mismo valor de laboratorio, en entrenamiento y en prueba a la vez; sumar el hierro del concentrado le entrega al modelo el gemelo de la respuesta. Mezclar horas sigue juntando vecinas con casi la misma sílice. Solo el orden temporal hace la pregunta que le importa a una planta, y ahí el modelo pierde contra el promedio en tres de cinco folds.

## 2. Qué se puede predecir

Cinco folds expansivos en orden temporal, de 674 horas cada uno, con 24 horas entre entrenamiento y prueba para que la autocorrelación no cruce la frontera. Solo se entrena y se evalúa con horas medidas (3.114 evaluadas). Se supone que el resultado de laboratorio llega dos horas después de la muestra; con un laboratorio más rápido, la persistencia sería más difícil de superar.

| Modelo | RMSE (puntos de sílice) | Reducción del error vs. el promedio |
|---|---:|---:|
| Ridge (proceso + laboratorio) | 0,842 | +37,0% |
| CatBoost (proceso + laboratorio) | 0,905 | +27,3% |
| Persistencia (laboratorio de hace 2 h) | 0,932 | +22,8% |
| XGBoost (proceso + laboratorio) | 0,949 | +20,1% |
| Ridge (proceso) | 1,040 | +4,0% |
| Promedio del entrenamiento | 1,061 | +0,0% |
| CatBoost (proceso) | 1,193 | -26,4% |
| XGBoost (proceso) | 1,197 | -27,3% |

![Resultados walk-forward](outputs/figures/walk_forward.png)

Cada fold: los modelos con resultados recientes de laboratorio (azul) quedan bajo la persistencia (naranjo); los modelos solo con proceso se dispersan alrededor del promedio, y los de árboles llegan a un RMSE de 1,53 a 1,59 en el segundo fold.

Un modelo lineal probablemente gana porque la señal útil es sobre todo "dónde estaba la sílice hace dos horas, ajustado por algunas variables de proceso"; los árboles gastan su capacidad en patrones de proceso que no se repiten de un mes a otro.

## 3. Qué se puede recetar

La primera versión de este proyecto recomendaba ajustes de reactivos y pH sobre una planta simulada. Acá la misma idea se prueba con datos reales. La prueba no es si el optimizador converge, sino si modelos entrenados con tramos distintos del tiempo coinciden en el movimiento.

Para las 40 horas con más sílice del último 20% de los datos (810 horas, unas cinco semanas), cuatro modelos sustitutos (XGBoost con las features de proceso, entrenados con el primer 25%, 50%, 75% y 100% de la historia anterior a esas horas) se optimizan sobre almidón, amina, pH y densidad de pulpa. Los movimientos se limitan a 10% del valor actual y al rango entre los percentiles 5 y 95 de lo que la planta operó alguna vez, para no pedirle a ningún modelo que extrapole lejos.

![Movimientos recomendados por cuatro modelos](outputs/figures/recommendations.png)

Cada modelo está seguro (una baja predicha de 0,43, 0,28, 0,33 y 0,33 puntos de sílice), y los modelos no se ponen de acuerdo en cómo lograrla. Los cuatro coinciden en la dirección del movimiento en el 35% de las horas para el almidón, el 52% para la amina, el 35% para el pH y el 10% para la densidad de pulpa. Un algoritmo genético (DEAP) y la evolución diferencial (SciPy) llegan al mismo óptimo sobre el mismo modelo (diferencia mediana de 0,00 puntos, máxima de 0,17), así que el desacuerdo es de los modelos, no de los optimizadores.

![Dosis de amina contra la sílice previa](outputs/figures/operator_reaction.png)

La razón más probable: la dosis sigue a la sílice. El flujo medio de amina sube con la sílice medida dos horas antes, hasta cerca de 3,5% de sílice, así que en los datos la amina alta va con la sílice alta y todos los modelos recomiendan menos. En la flotación catiónica inversa la amina es el colector que hace flotar el cuarzo; con menos, la sílice debería subir, no bajar. Un modelo ajustado a la operación histórica aprende cómo reaccionaban los operadores, no qué hacen los reactivos. Elegir setpoints requiere datos donde las dosis se movieron a propósito (pruebas en planta o tests escalón), no un mejor optimizador.

## Qué cambió respecto de la primera versión

La primera versión predecía la recuperación de cobre y molibdeno en 50.000 bloques de mineral simulados y optimizaba los reactivos con un algoritmo genético, NSGA-II y evolución diferencial, más una simulación de planta, SHAP y un servicio FastAPI. Cada una de sus cifras salía de un generador hecho para ella. Ahora corre con datos reales de planta, y el resultado invirtió el mensaje del proyecto: la predicción funciona solo con resultados recientes de laboratorio, y los datos observacionales no permiten recetar. El modelo de bloques, la simulación, el modelo multisalida, la comparación de deep learning y la API ya no están (siguen en el historial de git); los dos optimizadores se quedan, ahora como el control cruzado descrito arriba. El repositorio dejó de llamarse `optimizacion-geometalurgica-flotacion-cobre`, porque los datos no son de cobre; los enlaces antiguos en GitHub redirigen acá.

## Stack tecnológico

| Capa | Tecnología | Rol |
|---|---|---|
| Datos | **API de Kaggle**, **Polars** | Descarga, agregación horaria, interpolaciones y huecos |
| Modelos | **scikit-learn** (Ridge), **XGBoost**, **CatBoost** | Sensores virtuales con dos conjuntos de features |
| Estadística | **statsmodels** | Tests de Diebold-Mariano con varianza de Newey-West |
| Optimización | **SciPy** (`differential_evolution`), **DEAP** (algoritmo genético) | Búsqueda de setpoints dentro de una región de confianza |

## Cómo correrlo

Se necesita una vez una clave de la API de Kaggle configurada para su CLI (en `~/.kaggle/`, nunca en el repositorio), para descargar los datos.

```powershell
py -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/python main.py --download   # una vez: el CSV de 184 MB en data/raw/
./.venv/Scripts/python main.py              # unos minutos en la CPU de un notebook
```

Escribe `outputs/results.json`, la fuente de cada cifra de este README, y los gráficos en `outputs/figures/`.

### Tests

```powershell
./.venv/Scripts/pytest -v
```

Los tests corren sin red ni datos de Kaggle: decimales con coma, la detección de horas interpoladas, la antigüedad del ensayo de alimentación, rezagos de laboratorio que nunca cruzan el hueco de marzo ni presentan una hora interpolada como resultado de laboratorio, la evaluación walk-forward solo con horas medidas, Diebold-Mariano, la región de confianza y los dos optimizadores, el chequeo de reacción de los operadores y un chequeo de que cada cifra de la tabla de resultados de los dos README coincide con `outputs/results.json`.

## Licencia

Código: MIT, ver [LICENSE](LICENSE). Datos: CC0-1.0, no se redistribuyen acá.

## Autor

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
