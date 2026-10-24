# Smart Traffic Light Control — Etapa 1: Agrupar patrones de tráfico

> *Unsupervised learning clusters traffic patterns by time of day to know what kind of moment the intersection is in.*

Santiago · Erick · León · Edward

## ¿Qué hace?

Toma conteos horarios de vehículos por intersección y, con **K-Means** (aprendizaje no supervisado), descubre los "momentos" típicos de la intersección: **Valle**, **Normal**, **Pico**, etc. El resultado (modelo + datos etiquetados) es la entrada de las etapas 2 (predecir congestión) y 3 (decidir el verde con RL).

## Estructura (MVC + capa de servicios)

```
smart_traffic/
├── main.py                       # Punto de entrada (CLI): train | predict | web
├── server.py                     # Servidor Flask de desarrollo (python server.py)
├── wsgi.py                       # Punto de entrada WSGI para Gunicorn
├── gunicorn.conf.py              # Configuración de Gunicorn (producción)
├── config.py                     # Rutas, columnas, hiperparámetros, pesos
├── requirements.txt
├── app/
│   ├── web.py                    # Fábrica de la app Flask (create_app)
│   ├── models/                   # MODEL: datos y modelo de ML
│   │   ├── traffic_record.py     #   TrafficRecord, TrafficMoment (entidades)
│   │   └── clustering_model.py   #   TrafficPatternModel (scaler + KMeans, save/load)
│   ├── services/                 # SERVICE: lógica de negocio
│   │   ├── data_service.py       #   Carga Kaggle / Bogotá CGT → esquema estándar
│   │   ├── feature_service.py    #   Hora cíclica, día, carga relativa, tendencia
│   │   └── clustering_service.py #   Elige k, entrena, nombra clusters, clasifica
│   ├── controllers/              # CONTROLLER: orquesta el flujo
│   │   ├── clustering_controller.py  # Flujo train / predict (lo usan CLI y web)
│   │   └── web_controller.py     #   Rutas Flask: página + API JSON
│   └── views/                    # VIEW: presentación
│       ├── console_view.py       #   Tablas y métricas en consola
│       ├── plot_view.py          #   Gráficas PNG para el informe
│       ├── templates/index.html  #   Página web
│       └── static/               #   css/styles.css, js/app.js
├── scripts/generate_sample_data.py  # Datos sintéticos para probar
├── data/raw/                     # Aquí van los CSV descargados
└── outputs/                      # Modelo, datos etiquetados y gráficas
```

Flujo en consola: `main.py` → **ClusteringController** → **Services** (usan los **Models**) → **ConsoleView / PlotView**.

Flujo web: navegador → **WebController** (Flask) → **ClusteringController** → **Services** → JSON → `index.html` + `app.js`.

## Página web (Flask + Gunicorn)

**Desarrollo (Windows, Mac o Linux):**

```bash
python server.py          # o: python main.py web
```

Abre http://127.0.0.1:5000

**Producción con Gunicorn (Linux / macOS / WSL):**

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```

Abre http://localhost:8000. Se puede cambiar con variables de entorno: `GUNICORN_BIND`, `GUNICORN_WORKERS`, `GUNICORN_THREADS`, `GUNICORN_TIMEOUT`.

> Gunicorn no funciona en Windows. En Windows usen `python server.py`, o WSL si necesitan probar Gunicorn.

Desde la página se puede:
1. **Subir** el CSV de Kaggle o de Bogotá (queda guardado en `data/raw/`).
2. **Entrenar** el clustering eligiendo la fuente y k (automático o fijo).
3. **Ver** las métricas, la tabla de momentos y las 4 gráficas.
4. **Clasificar** el momento de una intersección para una fecha, hora y conteo de vehículos.

### API

| Método | Ruta | Para qué |
|---|---|---|
| GET | `/` | Página principal |
| GET | `/api/status` | Estado del modelo, datasets y último entrenamiento |
| POST | `/api/upload` | Subir CSV (form-data: `source`, `file`) |
| POST | `/api/train` | Entrenar (`{"source": "kaggle", "k": "auto"}`) |
| POST | `/api/predict` | Clasificar (`{"junction": "J1", "datetime": "2017-06-30T08:00", "vehicles": 90, "prev": "40, 70"}`) |
| GET | `/plots/<archivo>` | Gráficas generadas |

## Instalación

```bash
pip install -r requirements.txt
```

## Datos

| Fuente | Archivo esperado |
|---|---|
| Kaggle: Traffic Prediction Dataset | `data/raw/traffic.csv` (columnas `DateTime, Junction, Vehicles, ID`) |
| Bogotá Datos Abiertos: Conteo Vehículos CGT | `data/raw/conteo_vehiculos_cgt.csv` |
| Bogotá Datos Abiertos: Red semafórica | (se usa en etapas posteriores para la ubicación) |

Para el archivo de Bogotá las columnas se detectan automáticamente (fecha, hora, sensor, conteo). Si el archivo usa otros nombres, agrégalos en `config.BOGOTA_COLUMN_CANDIDATES`. Los conteos por minuto o cada 15 min se suman a totales por hora.

¿Sin datos aún? `python scripts/generate_sample_data.py` crea un `traffic.csv` sintético con el formato de Kaggle.

## Uso

```bash
# Entrenar (k automático por silhouette)
python main.py train
python main.py train --source bogota
python main.py train --source both --k 4
python main.py train --data otra/ruta.csv

# Clasificar el momento actual de una intersección
python main.py predict --junction J1 --datetime "2017-06-30 08:00" --vehicles 90 --prev 40 70
```

`--prev` son los conteos de las horas anteriores (de más antigua a más reciente); sirven para calcular la tendencia.

## Features que usa el clustering

| Feature | Por qué |
|---|---|
| `hour_sin`, `hour_cos` | Hora del día cíclica (las 23h están cerca de las 0h) |
| `dow_sin`, `dow_cos`, `is_weekend` | Día de la semana |
| `relative_load` | Vehículos ÷ promedio de esa intersección (compara intersecciones grandes y pequeñas) |
| `rolling_load` | Carga relativa media de las últimas 3 h |
| `trend` | ¿El tráfico sube o baja respecto a la hora anterior? |

Cada feature se estandariza y se multiplica por un peso (`config.FEATURE_WEIGHTS`) para que la carga de tráfico y la hora pesen más que el día de la semana.

## Salidas

- `outputs/models/traffic_pattern_model.joblib` — modelo para reutilizar en la etapa 2.
- `outputs/labeled_traffic.csv` — cada hora con su `cluster` y `moment` (feature para la etapa 2).
- `outputs/plots/` — selección de k, momentos por hora, mapa semanal y clusters en 2D (PCA).
- `outputs/training_summary.json` — resumen del último entrenamiento (lo lee la página).

## Uso desde otra etapa

```python
from app.models import TrafficPatternModel
from app.services import ClusteringService
import config

model = TrafficPatternModel.load(config.MODEL_FILE)
moment = ClusteringService().classify(model, "J1", datetime(2017, 6, 30, 8), 90, [40, 70])
print(moment.name)  # "Pico"
```
