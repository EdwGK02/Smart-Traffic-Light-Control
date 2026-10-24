"""
Modelo de clustering de patrones de tráfico.

Envuelve: escalador + KMeans + metadatos (nombres de los momentos, promedio por
intersección). Se guarda/carga como un solo archivo para usarlo en las etapas 2 y 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class TrafficPatternModel:
    """KMeans sobre features temporales + carga relativa de tráfico."""

    def __init__(self, n_clusters: int, random_state: int = 42,
                 weights: Optional[Dict[str, float]] = None):
        self.n_clusters = n_clusters
        self.weights = weights or {}
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        self.feature_names: List[str] = []
        self.cluster_names: Dict[int, str] = {}
        self.intersection_baseline: Dict[str, float] = {}  # promedio de vehículos por intersección
        self.is_fitted = False

    # ------------------------------------------------------------------ train
    def fit(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names = list(X.columns)
        self.scaler.fit(X.values)
        self.is_fitted = True
        labels = self.kmeans.fit_predict(self.scaled(X))
        return labels

    # ---------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.kmeans.predict(self.scaled(X))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Distancia de cada fila a cada centroide."""
        self._check_fitted()
        return self.kmeans.transform(self.scaled(X))

    def scaled(self, X: pd.DataFrame) -> np.ndarray:
        """Estandariza y aplica el peso de cada feature."""
        w = np.array([self.weights.get(f, 1.0) for f in self.feature_names])
        return self.scaler.transform(X[self.feature_names].values) * w

    def name_of(self, cluster_id: int) -> str:
        return self.cluster_names.get(int(cluster_id), f"Cluster {cluster_id}")

    @property
    def inertia(self) -> float:
        return float(self.kmeans.inertia_)

    # ---------------------------------------------------------- persistencia
    def save(self, path: Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "TrafficPatternModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el modelo en {path}. Entrénalo primero con: python main.py train"
            )
        return joblib.load(path)

    def baseline_for(self, intersection_id: str) -> Optional[float]:
        return self.intersection_baseline.get(str(intersection_id))

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado.")
