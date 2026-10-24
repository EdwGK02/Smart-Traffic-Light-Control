"""
Genera un CSV SINTÉTICO con el mismo formato del dataset de Kaggle
(DateTime, Junction, Vehicles, ID) para probar el código sin descargar nada.
Para el proyecto real usa el archivo de Kaggle.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def generate(path: Path = config.KAGGLE_FILE, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    junction_scale = {1: 45, 2: 15, 3: 13, 4: 7}
    for j, scale in junction_scale.items():
        start = "2015-11-01" if j != 4 else "2017-01-01"
        idx = pd.date_range(start, "2017-06-30 23:00", freq="h")
        h = idx.hour.values
        wk = idx.dayofweek.values >= 5
        # Doble pico (mañana y tarde) entre semana; curva suave el fin de semana
        weekday = 0.3 + 1.1 * np.exp(-((h - 8) ** 2) / 4) + 1.0 * np.exp(-((h - 18) ** 2) / 5) \
                  + 0.5 * np.exp(-((h - 13) ** 2) / 6)
        weekend = 0.3 + 0.8 * np.exp(-((h - 14) ** 2) / 12)
        shape = np.where(wk, weekend, weekday)
        growth = 1 + 0.4 * (np.arange(len(idx)) / len(idx))
        veh = np.maximum(1, rng.poisson(scale * shape * growth))
        for t, v in zip(idx, veh):
            rows.append((t, j, int(v), int(f"{t:%Y%m%d%H}{j}")))
    df = pd.DataFrame(rows, columns=["DateTime", "Junction", "Vehicles", "ID"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✔ Datos sintéticos: {len(df):,} filas → {path}")
    return path


if __name__ == "__main__":
    generate()
