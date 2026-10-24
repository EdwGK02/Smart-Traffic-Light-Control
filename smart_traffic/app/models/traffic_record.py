"""
Entidades del dominio.

Todas las fuentes (Kaggle, Bogotá CGT, ...) se convierten a un mismo esquema
estándar para que el resto del sistema no dependa del origen de los datos.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Esquema estándar de un DataFrame de tráfico (una fila = una intersección en una hora)
STANDARD_COLUMNS = ["timestamp", "intersection_id", "vehicles", "source"]


@dataclass
class TrafficRecord:
    """Conteo de vehículos de una intersección en una hora."""
    timestamp: datetime
    intersection_id: str
    vehicles: float
    source: str = "unknown"


@dataclass
class TrafficMoment:
    """Resultado de la etapa 1: en qué 'momento' está la intersección."""
    cluster_id: int
    name: str                      # ej. "Pico", "Valle"
    intersection_id: str
    timestamp: datetime
    vehicles: float
    relative_load: float           # vehículos / promedio de esa intersección
    distances: Optional[dict] = field(default=None)  # distancia a cada centroide

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "moment": self.name,
            "intersection_id": self.intersection_id,
            "timestamp": self.timestamp.isoformat(),
            "vehicles": self.vehicles,
            "relative_load": round(self.relative_load, 3),
        }
