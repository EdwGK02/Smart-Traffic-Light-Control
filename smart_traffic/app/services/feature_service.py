"""
FeatureService: convierte conteos horarios en features para el clustering.

Idea: un "momento" de la intersección depende de
  1) la hora del día y el día de la semana (codificados de forma cíclica),
  2) cuánta carga tiene comparada con lo normal en ESA intersección
     (así una intersección grande y una pequeña son comparables),
  3) la tendencia reciente (¿el tráfico está subiendo o bajando?).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config

FEATURES = [
    "hour_sin", "hour_cos",      # hora del día (cíclica: 23h está cerca de 0h)
    "dow_sin", "dow_cos",        # día de la semana (cíclico)
    "is_weekend",
    "relative_load",             # vehículos / promedio de la intersección
    "rolling_load",              # promedio de la carga relativa de las últimas N horas
    "trend",                     # cambio vs. la hora anterior
]


class FeatureService:

    def __init__(self, window: int = config.ROLLING_WINDOW_HOURS):
        self.window = window

    def build(self, df: pd.DataFrame,
              baseline: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Devuelve (df_con_features, baseline). Si baseline es None se calcula
        (entrenamiento); si se pasa, se reutiliza (predicción).
        """
        df = df.sort_values(["intersection_id", "timestamp"]).copy()

        if baseline is None:
            baseline = df.groupby("intersection_id")["vehicles"].mean().to_dict()
        global_mean = float(np.mean(list(baseline.values()))) if baseline else 1.0
        base = df["intersection_id"].map(baseline).fillna(global_mean).replace(0, 1.0)

        ts = df["timestamp"]
        hour = ts.dt.hour + ts.dt.minute / 60
        dow = ts.dt.dayofweek

        df["hour"] = ts.dt.hour
        df["day_of_week"] = dow
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["is_weekend"] = (dow >= 5).astype(int)
        df["relative_load"] = df["vehicles"] / base

        g = df.groupby("intersection_id")["relative_load"]
        df["rolling_load"] = g.transform(lambda s: s.rolling(self.window, min_periods=1).mean())
        df["trend"] = g.diff().fillna(0.0)

        return df, baseline

    @staticmethod
    def matrix(df: pd.DataFrame) -> pd.DataFrame:
        return df[FEATURES]
