"""
DataService: carga los datasets y los convierte al esquema estándar
[timestamp, intersection_id, vehicles, source], agregados por hora.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config
from app.models import STANDARD_COLUMNS


def _normalize(text: str) -> str:
    """Minúsculas, sin tildes ni espacios, para comparar nombres de columnas."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.strip().lower().replace(" ", "_")


class DataService:

    # ---------------------------------------------------------------- Kaggle
    def load_kaggle(self, path: Path = config.KAGGLE_FILE) -> pd.DataFrame:
        """Traffic Prediction Dataset (Kaggle): DateTime, Junction, Vehicles, ID."""
        path = Path(path)
        self._require(path, "Kaggle Traffic Prediction Dataset (traffic.csv)")
        raw = pd.read_csv(path)
        cols = config.KAGGLE_COLUMNS
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw[cols["timestamp"]]),
            "intersection_id": "J" + raw[cols["intersection_id"]].astype(str),
            "vehicles": pd.to_numeric(raw[cols["vehicles"]], errors="coerce"),
            "source": "kaggle",
        })
        return self._finalize(df)

    # ---------------------------------------------------------------- Bogotá
    def load_bogota(self, path: Path = config.BOGOTA_FILE,
                    column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Conteo Vehículos CGT (Secretaría Distrital de Movilidad).
        Detecta las columnas automáticamente; si falla, pasa column_map con
        las claves: timestamp, intersection_id, vehicles (y opcional time).
        Los conteos por minuto/15 min se suman a totales por hora.
        """
        path = Path(path)
        self._require(path, "Bogotá Conteo Vehículos CGT")
        raw = self._read_any(path)
        mapping = column_map or self._detect_columns(raw.columns, config.BOGOTA_COLUMN_CANDIDATES)

        missing = [k for k in ("timestamp", "intersection_id", "vehicles") if k not in mapping]
        if missing:
            raise ValueError(
                f"No se pudieron detectar las columnas {missing} en {path.name}.\n"
                f"Columnas del archivo: {list(raw.columns)}\n"
                "Agrega los nombres en config.BOGOTA_COLUMN_CANDIDATES o usa --map."
            )

        ts = raw[mapping["timestamp"]].astype(str)
        if "time" in mapping and mapping["time"] != mapping["timestamp"]:
            ts = ts + " " + raw[mapping["time"]].astype(str)

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(ts, errors="coerce", dayfirst=True),
            "intersection_id": "BOG_" + raw[mapping["intersection_id"]].astype(str),
            "vehicles": pd.to_numeric(raw[mapping["vehicles"]], errors="coerce"),
            "source": "bogota_cgt",
        })
        return self._finalize(df)

    # ------------------------------------------------------------- combinado
    def load(self, source: str, path: Optional[Path] = None, **kw) -> pd.DataFrame:
        if source == "kaggle":
            return self.load_kaggle(path or config.KAGGLE_FILE)
        if source == "bogota":
            return self.load_bogota(path or config.BOGOTA_FILE, **kw)
        if source == "both":
            frames = [self.load_kaggle(config.KAGGLE_FILE), self.load_bogota(config.BOGOTA_FILE, **kw)]
            return pd.concat(frames, ignore_index=True)
        raise ValueError(f"Fuente desconocida: {source}. Usa kaggle | bogota | both")

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y agrega por hora e intersección."""
        df = df.dropna(subset=["timestamp", "vehicles"])
        df = df[df["vehicles"] >= 0]
        df["timestamp"] = df["timestamp"].dt.floor("h")
        df = (df.groupby(["intersection_id", "timestamp", "source"], as_index=False)["vehicles"]
                .sum())
        return df[STANDARD_COLUMNS].sort_values(["intersection_id", "timestamp"]).reset_index(drop=True)

    @staticmethod
    def _detect_columns(columns, candidates: Dict[str, List[str]]) -> Dict[str, str]:
        norm = {_normalize(c): c for c in columns}
        found = {}
        for key, options in candidates.items():
            for opt in options:
                if _normalize(opt) in norm and norm[_normalize(opt)] not in found.values():
                    found[key] = norm[_normalize(opt)]
                    break
        return found

    @staticmethod
    def _read_any(path: Path) -> pd.DataFrame:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        # Los CSV de datos abiertos suelen venir con ; o , y en latin-1 o utf-8
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"No se pudo leer {path}")

    @staticmethod
    def _require(path: Path, name: str):
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}.\nDescarga '{name}' y guárdalo en esa ruta "
                "(o pasa --data <ruta>). Para probar sin datos reales: "
                "python scripts/generate_sample_data.py"
            )
