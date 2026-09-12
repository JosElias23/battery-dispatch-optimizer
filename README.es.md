# Optimización del despacho de una batería con precios reales de electricidad

[![CI](https://github.com/JosElias23/battery-dispatch-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/battery-dispatch-optimizer/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)](https://github.com/JosElias23/battery-dispatch-optimizer/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) · **Español**

¿Cuánto vale una batería conectada a la red, y qué parte de ese valor se puede
capturar realmente sin conocer los precios de mañana?

Un activo de almacenamiento de 100 MWh / 25 MW se despacha contra los precios
day-ahead alemanes de todo 2024 usando optimización entera mixta sobre precios
pronosticados, y se puntúa contra una referencia de previsión perfecta: lo que
gana el mismo optimizador cuando se le entregan los precios realizados.

---

## Resultados

Ingreso neto durante 2024 (8.784 horas), después de pérdidas round-trip y
degradación.

| Política | Ingreso neto | Capture rate | Ciclos/día |
|---|---:|---:|---:|
| Horario fijo (cargar de noche, vender en punta) | € 1.015.207 | 39,5 % | 0,93 |
| Umbral de precio (regla por cuantil) | € 951.999 | 37,0 % | 0,67 |
| MILP + forecast seasonal-naive | € 2.154.597 | 83,7 % | 1,42 |
| **MILP + forecast con gradient boosting** | **€ 2.205.860** | **85,7 %** | 1,50 |
| _Previsión perfecta (cota teórica)_ | _€ 2.573.659_ | _100 %_ | 1,45 |

![Capture rates](reports/figures/capture_rates.png)

**El titular: optimizar contra un forecast captura el 85,7 % de lo que habría
ganado un operador clarividente, y gana € 1,19 M más que un horario fijo, 2,17×
el ingreso, con el mismo activo físico.**

El capture rate, y no la cifra en euros, es el número que vale la pena citar. El
ingreso absoluto depende por completo de lo volátil que haya resultado ser el
año; la fracción del óptimo alcanzable sí es comparable entre mercados y años.

Cada número acá lo produce `scripts/run_experiment.py` y queda guardado en
[`reports/metrics_dispatch.json`](reports/metrics_dispatch.json). Nada en este
README está escrito a mano.

### Qué hace realmente el optimizador

![Semana de ejemplo](reports/figures/example_week.png)

La semana más volátil de 2024. Los precios llegan a 936 €/MWh; la batería cicla
entre sus límites de estado de carga de 10 % y 90 %, comprando en los valles y
vendiendo en los peaks, y deja € 110.769 en siete días.

---

## Cinco hallazgos

### 1. Un forecast mejor en promedio no vale automáticamente más dinero

| Pronosticador | MAE (€/MWh) | RMSE (€/MWh) | Ingreso | Capture |
|---|---:|---:|---:|---:|
| Seasonal naive (misma hora, semana anterior) | 32,31 | 54,78 | € 2.154.597 | 83,7 % |
| Gradient boosting | **28,36** | **43,94** | **€ 2.205.860** | **85,7 %** |

Una reducción de 12 % en el MAE compra 2,0 puntos porcentuales de capture rate:
real, pero lejos de proporcional. El despacho no necesita precios exactos;
necesita el *orden* entre horas baratas y caras. Equivocarse en el nivel de una
tarde plana no cuesta nada. Equivocarse en la posición del peak de la tarde
cuesta un ciclo completo.

El resultado intermedio lo deja más claro. Una versión anterior del booster
tenía un MAE *peor* que el baseline naive (25,97 vs 24,77 en un tramo de dos
meses) y, aun así, el arreglo que importaba subió el capture de 44,5 % a 55,1 %
en ese mismo tramo moviendo el MAE apenas nada. **El error de forecast es una
métrica proxy. El ingreso es el objetivo.** Acá se reportan ambos; no coinciden.

### 2. Entrenar sobre *niveles* de precio rompió el modelo, y centrar lo arregló

El primer gradient booster se entrenó para predecir el precio directamente.
Aprendió 2023, donde los precios promediaron 95,18 €/MWh. Se evaluó sobre 2024,
que cerró en 78,51 €/MWh, 17 €/MWh más abajo. Toda su noción de «barato» quedó
en el lugar equivocado, y perdió contra un lag semanal ingenuo.

El arreglo fue predecir la *desviación respecto de un nivel de precio móvil de
168 horas* en vez del nivel mismo, con las features rezagadas centradas en el
mismo ancla. La **forma** diaria y semanal de los precios eléctricos es estable
entre años; el nivel absoluto no lo es, y el nivel fue justamente lo que se
movió.

Este es un cambio de distribución que una partición train/test dentro de un solo
año nunca habría dejado a la vista. Solo se ve porque la partición es entre
años, que es la única partición que refleja cómo se desplegaría el modelo en la
realidad.

### 3. La restricción que todos llaman redundante no lo es, pero es barata, no lucrativa

Las formulaciones de manual suelen eliminar la binaria que prohíbe cargar y
descargar al mismo tiempo, con el argumento de que hacer ambas cosas a la vez
nunca es óptimo, de modo que la relajación lineal es exacta y mucho más rápida.

Ese argumento supone precios positivos. 457 horas de 2024, el 5,2 % del año,
cerraron **bajo cero**, y ahí al operador le *pagan* por consumir. El modelo
relajado descubre entonces que puede mantener plano el estado de carga mientras
sostiene una importación neta:

```
hold SoC:   eta_c * charge = discharge / eta_d
net grid:   discharge - charge = charge * (eta_rt - 1) < 0
```

La pérdida round-trip se convierte en una forma de absorber indefinidamente
energía por la que te pagan, sin llenar nunca la batería. Ejecutando
`scripts/ablate_complementarity.py`:

| | Con la restricción | Sin ella |
|---|---:|---:|
| Ingreso neto | € 2.573.659 | € 2.574.867 |
| Horas cargando y descargando a la vez | 0 | **28** |
| ...de las cuales a precios negativos | — | **28 (100 %)** |
| Horas físicamente imposibles | 0 | 28 |

El caso más extremo carga a 25 MW mientras descarga 11,8 MW a −85,08 €/MWh.

La conclusión honesta es sobre corrección, no sobre dinero. Cada una de las 28
horas imposibles ocurre a precio negativo, exactamente como predice el
mecanismo. Pero el ingreso inventado es € 1.209, o un **0,05 %**. La relajación
produce un programa físicamente imposible y apenas se enriquece con eso.

La relajación tampoco es la aceleración dramática que sugiere el argumento de
manual: resolver el año completo toma 4,43 s con la binaria y 2,80 s sin ella,
un factor de 1,6. El branch-and-bound moderno maneja bien esta estructura.

Entonces: mantener la binaria. Cuesta 1,6 s al año y es la diferencia entre un
programa que el activo podría ejecutar y uno que no.

### 4. Llegar al último 14 % es un problema de forecasting, no de optimización

El MILP es óptimo para los precios que recibe, así que la brecha de 14,3 %
debería ser error de forecast. Una versión anterior de este README lo afirmaba
—"hasta el último euro", y que "ninguna mejora del solver, de la formulación o
del horizonte puede recuperar nada de eso"—. Eso era un argumento, no una
medición, y comparaba dos brazos que difieren en **dos** cosas: la referencia
optimiza con precios reales y además planifica sobre el año completo en vez de
un horizonte rodante de 48 horas.

`scripts/decompose_gap.py` agrega el brazo que los separa: el mismo horizonte
rodante, la misma ventana de 48 horas y el mismo compromiso de 24, con los
precios realizados.

| Brazo | Ingreso neto | Brecha a la referencia |
|---|---:|---:|
| Previsión perfecta, trozos de catorce días | € 2.573.659 | — |
| **Rodante 48 h, precios reales** | **€ 2.574.513** | **−€ 855** |
| Rodante 48 h, pronóstico GB | € 2.205.860 | € 367.799 |

**El horizonte no cuesta nada: −0,2 % de la brecha, con el 100,2 % restante en
error de forecast.** La afirmación se sostiene, y ahora está medida. La razón
está en el activo y no en el solver: una batería de 100 MWh / 25 MW tiene cuatro
horas de duración, así que se llena y se vacía dentro del día y nunca necesita
mover energía a través de una semana. Un horizonte del doble del ciclo ya es
suficientemente largo.

De ahí salió otra cosa. El brazo rodante **supera** a la referencia por € 855,
así que la referencia no es una cota superior. `perfect_foresight_dispatch`
resuelve el año en trozos de catorce días, y una ventana de 48 horas ve a través
de las costuras que los trozos no pueden cruzar. El margen es 0,03 % y no mueve
ninguna conclusión, pero cambia de qué es fracción una tasa de captura: de una
referencia de previsión perfecta por trozos, no de lo máximo que un operador
podría haber ganado.

### 5. El ingreso es notablemente insensible a *cuándo* caen los errores de forecast

![Monte Carlo](reports/figures/monte_carlo.png)

300 años simulados, cada uno construido remuestreando los errores de forecast
observados en 2024 en bloques de un día y volviendo a correr el despacho
completo con rolling horizon:

| | |
|---|---:|
| Ingreso anual medio | € 1.979.567 |
| Desviación estándar | € 24.533 |
| Intervalo de 95 % | € 1.923.083 – € 2.025.870 |
| **Dispersión relativa (1 sd)** | **1,24 % de la media** |

Una banda de ±1,24 % es angosta. Dado un pronosticador de esta calidad, el
*momento* en que caen sus errores casi no importa: la mala suerte en cuándo
ocurren cuesta cerca de € 25.000 sobre € 2 M. Para el dueño del activo, esa es
la diferencia entre un modelo de ingresos que se puede respaldar financieramente
y uno que no.

**La corrida real superó a la simulación, y eso hay que explicarlo en vez de
celebrarlo.** El despacho efectivo con gradient boosting ganó € 2.205.860, por
sobre el intervalo de 95 %. O tuvo suerte, o la simulación es pesimista por
construcción. `scripts/analyse_forecast_error.py` pone a prueba la segunda
hipótesis:

| Forecast | MAE (€/MWh) | Correlación de rangos media intradía |
|---|---:|---:|
| Gradient boosting real | 28,36 | **0,7757** |
| Bootstrap por bloques | 28,22 | **0,7266** (sd 0,0119) |

Misma magnitud de error, ranking claramente peor. El remuestreo por bloques
conserva qué tan *grandes* son los errores y cómo se agrupan en el tiempo, pero
los desprende de los precios frente a los que se cometieron. Un bloque de error
de una semana volátil de diciembre pegado sobre un día tranquilo de julio es un
forecast que ningún modelo habría producido.

Y el valor del despacho vive por completo en el ranking, no en el nivel: un
forecast uniformemente 30 €/MWh demasiado alto no pierde absolutamente nada,
porque el optimizador sigue comprando en las mismas horas. Así que el Monte
Carlo es una **cota conservadora**, y el resultado realizado es estructura que el
bootstrap descarta a propósito, no suerte.

Este es el hallazgo 1 llegando por una segunda vía independiente. Dos
experimentos distintos, la misma conclusión: *para el despacho de
almacenamiento, el orden es la métrica y la magnitud del error es una
distracción.*

En el camino apareció una complicación que conviene decir. El pronosticador es
**peor** justo donde está el dinero: el error absoluto medio es 27,63 €/MWh en
el cuartil de horas más tranquilas contra 34,18 €/MWh en el más volátil, 1,24×
más alto, correlación 0,36 con la volatilidad local. Las horas volátiles son
donde están los spreads. Mejorar el forecast específicamente en esas horas es la
vía más prometedora para cerrar el 14 % de brecha que queda.

Por qué se tomó cada una de estas decisiones, y cuánto costaban las
alternativas, está escrito en [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## El problema

Una batería gana dinero comprando electricidad cuando está barata y vendiéndola
cuando está cara. Tres cosas hacen eso más difícil de lo que suena:

- **Pérdidas round-trip.** Con 86 % de eficiencia, entregar 1 MWh a la red
  obliga a sacarle 1,163 MWh. Comprar a 100 €/MWh significa que el precio de
  venta debe superar 119,28 €/MWh, incluyendo degradación, solo para empatar.
- **Degradación.** Ciclar desgasta las celdas. Sin un costo sobre el throughput,
  el optimizador transa cada pequeña oscilación, lo que es rentable en el papel
  y destruye el activo en la realidad.
- **Incertidumbre.** Los precios de mañana no se conocen cuando se arma el plan
  de hoy.

El modelo físico, las convenciones de signo y la economía están en
[`src/battery/model.py`](src/battery/model.py).

## Los datos

Precios de la subasta day-ahead de Alemania/Luxemburgo vía la API de
[energy-charts](https://api.energy-charts.info) (Fraunhofer ISE), que republica
datos de Bundesnetzagentur/SMARD bajo **CC BY 4.0**. Sin API key, sin registro.

| | 2023 (entrenamiento) | 2024 (evaluación) |
|---|---:|---:|
| Horas | 8.760 | 8.784 |
| Precio medio | 95,18 €/MWh | 78,51 €/MWh |
| Desviación estándar | 47,58 €/MWh | 52,72 €/MWh |
| Rango | −500,00 a 524,27 | −135,45 a 936,28 |
| Horas con precio negativo | 301 (3,44 %) | **457 (5,20 %)** |
| Spread diario medio | 98,13 €/MWh | 111,47 €/MWh |

Los pronosticadores se ajustan sobre 2023 y nunca ven 2024. Ese spread diario es
la oportunidad de arbitraje en bruto: ningún ciclo completo puede ganar más que
el spread del día en que ocurre.

> **Por qué Alemania y no Chile.** La API del Coordinador Eléctrico Nacional
> exige una clave registrada (`Authentication parameters missing`), y el portal
> de datos abiertos de la CNE estuvo inaccesible durante el desarrollo. Exigir
> una credencial rompería la garantía de que cualquiera pueda clonar este
> repositorio y reproducir cada número. Alemania es además el mejor caso de
> prueba: una penetración renovable muy alta produce la volatilidad y los
> precios negativos que hacen interesante el arbitraje con almacenamiento.
> `src/battery/data.py` aísla todo el acceso a la red detrás de una sola
> función, así que agregar un adaptador chileno significa escribir una clase y
> no cambiar nada más.

## Método

### Formulación entera mixta

Por hora `t`, con precios `p[t]`:

```
maximise   sum_t  p[t] * (d[t] - c[t]) * dt  -  k_deg * sum_t d[t] * dt

subject to s[t] = s[t-1] + eta_c * c[t] * dt - d[t] * dt / eta_d   (balance de energía)
           0 <= c[t] <= P * y[t]                                   (potencia de carga)
           0 <= d[t] <= P * (1 - y[t])                             (potencia de descarga)
           SoC_min <= s[t] <= SoC_max                              (banda utilizable)
           y[t] in {0, 1}                                          (una sola dirección)
```

Se resuelve con HiGHS a través de PuLP, con fallback al CBC incluido para que el
repositorio corra sin dependencias adicionales del sistema. El año completo se
resuelve en unos 5 segundos.

### Rolling horizon

Un operador real no resuelve el año de una sola vez. Cada día el modelo
planifica 48 horas hacia adelante sobre precios pronosticados y **compromete
solo las primeras 24**, y al día siguiente vuelve a planificar.

Comprometer menos que el horizonte de planificación es lo que evita que el
programa colapse en el borde de la ventana: la energía que queda en la batería
al final de una ventana no vale nada *dentro* de esa ventana, así que un modelo
que compromete todo lo que planifica vacía la batería cada noche sin importar
los precios de mañana. El segundo día, que se descarta, absorbe el efecto de
borde.

Los planes se hacen sobre precios pronosticados; **el ingreso siempre se
contabiliza a precios realizados.** Evaluar un plan guiado por un forecast
contra el mismo forecast que lo produjo no mide nada más que la aritmética del
optimizador.

### Cómo se evita la fuga temporal

La hora más lejana que se está decidiendo está a 48 horas, así que **ninguna
feature puede referirse a nada dentro de las 48 horas previas a su objetivo**.
Eso descarta el predictor más fuerte disponible, el precio de ayer a la misma
hora, y `src/battery/forecast.py` lanza un error ante cualquier lag menor a 48.

Esa protección detectó de inmediato un lag de 24 que había quedado en la config
de un borrador temprano. Habría producido un mejor forecast, un mejor capture
rate y un ingreso que nunca se habría podido ganar.

`tests/test_forecast.py` va más lejos: sobrescribe la segunda mitad de la serie
de precios con basura y verifica que cada predicción de la primera mitad quede
idéntica bit a bit. Cualquier feature que mire hacia adelante en el tiempo falla
esa prueba.

### Verificación del solver

`check_feasibility` vuelve a derivar el estado de carga a partir de un programa
y revisa de nuevo cada límite físico, de forma independiente del solver. Que un
solver reporte `Optimal` significa únicamente que satisfizo las restricciones
que le entregaron; si esas estaban mal escritas, la respuesta es óptima para el
problema equivocado, y el error aparece como ingreso en vez de como excepción.
Cada despacho de la tabla de resultados pasa esta verificación antes de que se
registre su ingreso.

---

## Cómo reproducir estos números

```bash
git clone https://github.com/JosElias23/battery-dispatch-optimizer.git
cd battery-dispatch-optimizer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

```bash
python -m pytest
```

69 pruebas que cubren la física de la batería, la optimalidad del MILP contra
casos calculados a mano, y la fuga de información en el forecast.

```bash
python scripts/run_experiment.py
```

Descarga y cachea los datos de precios, ajusta los pronosticadores sobre 2023,
despacha 2024 bajo cada política y escribe `reports/metrics_dispatch.json`.
Alrededor de un minuto.

```bash
python scripts/ablate_complementarity.py
python scripts/analyse_forecast_error.py
python scripts/decompose_gap.py
python scripts/make_figures.py
```

El Monte Carlo es el paso caro: un año simulado son 366 resoluciones MILP
secuenciales, así que 300 réplicas toman cerca de una hora repartidas en 12
procesos.

```bash
python scripts/run_monte_carlo.py --workers 12
```

O corre `make all` para el pipeline completo.

---

## Estructura del repositorio

```
battery-dispatch-optimizer/
├── configs/default.yaml          cada parámetro que mueve un número reportado
├── src/battery/
│   ├── model.py                  física, economía, chequeo de factibilidad independiente
│   ├── optimize.py               MILP y la cota de previsión perfecta
│   ├── policies.py               baselines por reglas y el simulador compartido
│   ├── forecast.py               pronosticadores con el bloqueo de fuga de 48 horas
│   ├── simulate.py               rolling horizon y Monte Carlo
│   ├── data.py                   descarga, caché y validación de precios
│   └── utils.py                  seeding, config, reporte en JSON
├── scripts/
│   ├── run_experiment.py         el experimento principal
│   ├── run_monte_carlo.py        distribución de ingresos bajo error de forecast
│   ├── ablate_complementarity.py la ablación de la bomba de dinero
│   ├── analyse_forecast_error.py por qué el Monte Carlo es una cota conservadora
│   ├── decompose_gap.py          costo de horizonte vs. error de forecast
│   └── make_figures.py           cada figura de este README
├── tests/                        69 pruebas
└── reports/                      métricas en JSON, figuras en PNG
```

---

## Limitaciones y próximos pasos

**Los precios day-ahead se publican antes de que empiece el día.** En el mercado
alemán real la subasta cierra alrededor de las 12:45 de D−1 y publica los 24
precios del día D, así que un operador que planifica *dentro* del día de entrega
efectivamente tiene una previsión casi perfecta. El problema de forecasting que
se modela acá es el que se enfrenta al **ofertar en** la subasta antes de que
cierre, y el bloqueo de 48 horas es una versión deliberadamente conservadora de
él. Una mesa que oferta al mediodía de D−1 enfrenta un horizonte de 12 a 36
horas, no de 48. Un bloqueo más corto subiría todos los capture rates reportados
arriba. El orden relativo de las políticas no cambiaría.

**Un año, un mercado, una configuración de batería.** Todos los resultados son
de 2024 en **Alemania con una batería de 4 horas.** Los capture rates dependen
de la volatilidad del año, de la duración del activo y de la estructura del
mercado. Nada de esto ha sido probado con datos chilenos.

**El ingreso es solo arbitraje.** Los activos de almacenamiento reales obtienen
una parte importante de sus ingresos de la respuesta en frecuencia y de los
mercados de capacidad, que acá no se modelan. Las cifras son una cota inferior
del valor total del activo y no deberían leerse como un caso de negocio.

**La degradación es lineal en el throughput.** El envejecimiento real de las
celdas depende de la profundidad de descarga, la temperatura, la tasa C y el
tiempo calendario. Un costo lineal por MWh es la aproximación tratable estándar
y mantiene el problema como un MILP representable con LP, pero valoriza mal los
ciclos profundos.

**La batería parte medio llena.** Esos 50 MWh iniciales se pueden vender sin
haberlos comprado nunca. En un año valen aproximadamente el 0,1 % del ingreso y
aplican por igual a todas las políticas, así que las comparaciones no se ven
afectadas, pero las cifras absolutas quedan levemente favorecidas.

**La previsión perfecta se resuelve en bloques de catorce días**, encadenados a
través del estado de carga, en vez de como un único MILP de 8.784 horas. Eso
solo puede *subestimar* el óptimo verdadero, porque prohíbe arbitrar entre
bloques, así que los capture rates reportados son, si acaso, levemente
generosos.

**Sin búsqueda de hiperparámetros.** El booster usa valores estándar aplicados
sin tuning. Justo como comparación, casi con certeza no óptimo.

**El Monte Carlo modela el momento del error, no su tamaño.** Remuestrea los
errores que el modelo ajustado efectivamente cometió, así que responde «qué
habría pasado si esos errores hubieran caído en otro momento del año», no «qué
habría pasado si el pronosticador fuera peor». Un análisis de sensibilidad a la
calidad del pronosticador, escalando la magnitud del error hacia arriba y hacia
abajo, respondería la segunda pregunta y acá no se hace.

### Planificado

- Adaptador para el mercado chileno, cuando haya disponible una API key del
  Coordinador
- Optimización estocástica (basada en escenarios) en vez de forecasts puntuales,
  que debería recuperar parte del 14 % de brecha que queda
- Ingresos por respuesta en frecuencia apilados sobre el arbitraje
- Sensibilidad del capture rate a la duración de la batería (2 h, 4 h, 8 h)

---

## Licencia

**MIT, ver [LICENSE](LICENSE).** Los datos de precios son de
Bundesnetzagentur/SMARD vía energy-charts de Fraunhofer ISE, licenciados
CC BY 4.0.
