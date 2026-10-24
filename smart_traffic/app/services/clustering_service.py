"""
ClusteringService: elige k, entrena el modelo, nombra los clusters y
clasifica nuevas observaciones en un 'momento' del semáforo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import config
from app.models import TrafficMoment, TrafficPatternModel
from app.services.feature_service import FeatureService


@dataclass
class TrainingResult:
    model: TrafficPatternModel
    data: pd.DataFrame                       # datos con features + columna 'cluster' y 'moment'
    k_scores: pd.DataFrame                   # k, inertia, silhouette
    metrics: Dict[str, float] = field(default_factory=dict)
    profiles: Optional[pd.DataFrame] = None  # resumen por cluster


class ClusteringService:

    def __init__(self, features: Optional[FeatureService] = None,
                 random_state: int = config.RANDOM_STATE):
        self.features = features or FeatureService()
        self.random_state = random_state

    # ------------------------------------------------------------- elegir k
    def evaluate_k(self, X: pd.DataFrame, k_range: Sequence[int] = config.K_RANGE) -> pd.DataFrame:
        w = np.array([config.FEATURE_WEIGHTS.get(c, 1.0) for c in X.columns])
        Xs = StandardScaler().fit_transform(X.values) * w
        sample = min(config.SILHOUETTE_SAMPLE, len(Xs))
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Xs)
            sil = silhouette_score(Xs, km.labels_, sample_size=sample, random_state=self.random_state)
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- train
    def train(self, df: pd.DataFrame, k: Optional[int] = None) -> TrainingResult:
        feat_df, baseline = self.features.build(df)
        X = self.features.matrix(feat_df)

        k_scores = self.evaluate_k(X)
        if k is None:
            k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])

        model = TrafficPatternModel(n_clusters=k, random_state=self.random_state,
                                    weights=config.FEATURE_WEIGHTS)
        labels = model.fit(X)
        model.intersection_baseline = {str(a): float(b) for a, b in baseline.items()}
        feat_df["cluster"] = labels

        model.cluster_names = self._name_clusters(feat_df, k)
        feat_df["moment"] = feat_df["cluster"].map(model.cluster_names)

        Xs = model.scaled(X)
        sample = min(config.SILHOUETTE_SAMPLE, len(Xs))
        metrics = {
            "k": k,
            "silhouette": float(silhouette_score(Xs, labels, sample_size=sample,
                                                 random_state=self.random_state)),
            "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(Xs, labels)),
            "inertia": model.inertia,
            "n_rows": int(len(feat_df)),
            "n_intersections": int(feat_df["intersection_id"].nunique()),
        }
        return TrainingResult(model=model, data=feat_df, k_scores=k_scores,
                              metrics=metrics, profiles=self.profile(feat_df))

    # ------------------------------------------------------------ nombrado
    @staticmethod
    def _name_clusters(feat_df: pd.DataFrame, k: int) -> Dict[int, str]:
        """Ordena los clusters por carga relativa media y les asigna un nombre."""
        order = feat_df.groupby("cluster")["relative_load"].mean().sort_values().index.tolist()
        names = config.MOMENT_NAMES.get(k, [f"Nivel {i + 1}" for i in range(k)])
        result = {int(cid): names[rank] for rank, cid in enumerate(order)}
        # Añade sufijo de fin de semana a clusters dominados por sábado/domingo
        wk = feat_df.groupby("cluster")["is_weekend"].mean()
        for cid, share in wk.items():
            if share > 0.6:
                result[int(cid)] += " (fin de semana)"
        return result

    # ------------------------------------------------------------ perfiles
    @staticmethod
    def profile(feat_df: pd.DataFrame) -> pd.DataFrame:
        """Resumen interpretable de cada cluster."""
        def hours_label(s: pd.Series) -> str:
            top = s.value_counts().head(4).index.sort_values()
            return ", ".join(f"{h:02d}h" for h in top)

        prof = feat_df.groupby(["cluster", "moment"]).agg(
            filas=("vehicles", "size"),
            vehiculos_prom=("vehicles", "mean"),
            carga_relativa=("relative_load", "mean"),
            tendencia=("trend", "mean"),
            pct_fin_semana=("is_weekend", "mean"),
            horas_tipicas=("hour", hours_label),
        ).reset_index()
        prof["pct_datos"] = prof["filas"] / prof["filas"].sum()
        return prof.sort_values("carga_relativa").reset_index(drop=True)

    # ------------------------------------------------------------ predicción
    def classify(self, model: TrafficPatternModel, intersection_id: str,
                 timestamp: datetime, vehicles: float,
                 previous_vehicles: Optional[List[float]] = None) -> TrafficMoment:
        """
        Clasifica una observación nueva. previous_vehicles = conteos de las horas
        anteriores (de más antigua a más reciente) para calcular tendencia.
        """
        previous_vehicles = previous_vehicles or []
        hist = [*previous_vehicles, vehicles]
        stamps = pd.date_range(end=pd.Timestamp(timestamp).floor("h"), periods=len(hist), freq="h")
        df = pd.DataFrame({
            "timestamp": stamps,
            "intersection_id": str(intersection_id),
            "vehicles": hist,
            "source": "live",
        })
        feat_df, _ = self.features.build(df, baseline=model.intersection_baseline)
        last = feat_df.tail(1)
        X = self.features.matrix(last)
        cid = int(model.predict(X)[0])
        dist = model.transform(X)[0]
        return TrafficMoment(
            cluster_id=cid,
            name=model.name_of(cid),
            intersection_id=str(intersection_id),
            timestamp=pd.Timestamp(timestamp).to_pydatetime(),
            vehicles=float(vehicles),
            relative_load=float(last["relative_load"].iloc[0]),
            distances={model.name_of(i): float(d) for i, d in enumerate(dist)},
        )
