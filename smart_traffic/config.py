"""
Configuración central del proyecto Smart Traffic Light Control.
Etapa 1: agrupar patrones de tráfico por hora del día (clustering no supervisado).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

for _d in (RAW_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Archivos por defecto (descárgalos y colócalos en data/raw/)
KAGGLE_FILE = RAW_DIR / "traffic.csv"            # Kaggle: Traffic Prediction Dataset
BOGOTA_FILE = RAW_DIR / "conteo_vehiculos_cgt.csv"  # Bogotá Datos Abiertos: Conteo Vehículos CGT
MODEL_FILE = MODELS_DIR / "traffic_pattern_model.joblib"
SUMMARY_FILE = OUTPUT_DIR / "training_summary.json"   # lo lee la página web

# Flask
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
MAX_UPLOAD_MB = 200

# ---------------------------------------------------------------------------
# Mapeo de columnas
# ---------------------------------------------------------------------------
# Kaggle (fedesoriano/traffic-prediction-dataset): DateTime, Junction, Vehicles, ID
KAGGLE_COLUMNS = {
    "timestamp": "DateTime",
    "intersection_id": "Junction",
    "vehicles": "Vehicles",
}

# Bogotá CGT: los nombres de columnas pueden cambiar entre versiones del archivo.
# El servicio prueba estos candidatos en orden (sin importar mayúsculas/tildes).
# Si tu archivo usa otros nombres, agrégalos aquí.
BOGOTA_COLUMN_CANDIDATES = {
    "timestamp": ["fecha_hora", "fechahora", "timestamp", "fecha", "date", "datetime"],
    "time": ["hora", "hour", "time", "intervalo", "hora_inicio"],
    "intersection_id": ["id_sensor", "sensor", "codigo_sensor", "id_dispositivo",
                        "dispositivo", "nombre_sensor", "id", "punto", "interseccion"],
    "vehicles": ["conteo", "total", "volumen", "cantidad", "vehiculos",
                 "conteo_vehiculos", "total_vehiculos", "count"],
}

# ---------------------------------------------------------------------------
# Hiperparámetros del clustering
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
K_RANGE = range(3, 8)          # valores de k que se prueban si k = "auto"
SILHOUETTE_SAMPLE = 8000       # muestra para calcular silhouette (rápido con 48k filas)
ROLLING_WINDOW_HOURS = 3       # ventana para la tendencia reciente

# Peso de cada feature después de estandarizar (más peso = influye más en el cluster).
# Queremos que el "momento" dependa sobre todo de la carga de tráfico y la hora.
FEATURE_WEIGHTS = {
    "hour_sin": 1.0, "hour_cos": 1.0,
    "dow_sin": 0.3, "dow_cos": 0.3,
    "is_weekend": 0.6,
    "relative_load": 2.0,
    "rolling_load": 1.5,
    "trend": 0.7,
}

# Nombres de los "momentos" del semáforo, de menor a mayor congestión.
# Se asignan automáticamente según el volumen medio de cada cluster.
MOMENT_NAMES = {
    3: ["Valle", "Normal", "Pico"],
    4: ["Valle", "Bajo", "Alto", "Pico"],
    5: ["Valle", "Bajo", "Normal", "Alto", "Pico"],
    6: ["Madrugada", "Valle", "Bajo", "Normal", "Alto", "Pico"],
    7: ["Madrugada", "Valle", "Bajo", "Normal", "Alto", "Pico", "Pico extremo"],
}
