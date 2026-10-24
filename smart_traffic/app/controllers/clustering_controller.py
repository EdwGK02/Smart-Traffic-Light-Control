"""
ClusteringController: orquesta el flujo de la Etapa 1.

  train   -> carga datos -> features -> KMeans -> guarda modelo -> muestra resultados
  predict -> carga modelo -> clasifica una observación -> muestra el momento
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config
from app.models import TrafficPatternModel
from app.services import ClusteringService, DataService
from app.views import ConsoleView, PlotView


class ClusteringController:

    def __init__(self, data_service: Optional[DataService] = None,
                 clustering_service: Optional[ClusteringService] = None,
                 console: Optional[ConsoleView] = None,
                 plots: Optional[PlotView] = None):
        self.data = data_service or DataService()
        self.clustering = clustering_service or ClusteringService()
        self.console = console or ConsoleView()
        self.plots = plots or PlotView()

    def train(self, source: str = "kaggle", data_path: Optional[Path] = None,
              k: Optional[int] = None, model_path: Path = config.MODEL_FILE,
              make_plots: bool = True):
        self.console.header("Smart Traffic Light Control · Etapa 1: agrupar patrones")
        self.console.info(f"Cargando datos ({source})...")
        df = self.data.load(source, data_path)
        self.console.data_summary(df)

        self.console.info("Entrenando clustering (probando k = "
                          f"{min(config.K_RANGE)}..{max(config.K_RANGE)})..."
                          if k is None else f"Entrenando clustering con k = {k}...")
        result = self.clustering.train(df, k=k)

        self.console.k_selection(result.k_scores, result.metrics["k"])
        self.console.metrics(result.metrics)
        self.console.profiles(result.profiles)

        saved = result.model.save(model_path)
        self.console.success(f"Modelo guardado en {saved}")

        # Datos etiquetados: los usará la Etapa 2 (predicción de congestión)
        labeled = config.OUTPUT_DIR / "labeled_traffic.csv"
        cols = ["timestamp", "intersection_id", "vehicles", "relative_load", "cluster", "moment"]
        result.data[cols].to_csv(labeled, index=False)
        self.console.success(f"Datos etiquetados guardados en {labeled}")

        plot_files = []
        if make_plots:
            plot_files = self.plots.render_all(result)
            self.console.success("Gráficas: " + ", ".join(str(f.name) for f in plot_files)
                                 + f"  (en {config.PLOTS_DIR})")

        self._save_summary(result, source, df, plot_files)
        return result

    @staticmethod
    def _save_summary(result, source, df, plot_files):
        """Guarda un resumen JSON del entrenamiento (lo usa la página web)."""
        prof = result.profiles.copy()
        summary = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "date_range": [f"{df['timestamp'].min():%Y-%m-%d}", f"{df['timestamp'].max():%Y-%m-%d}"],
            "metrics": result.metrics,
            "k_scores": result.k_scores.round(4).to_dict(orient="records"),
            "profiles": prof.round(4).to_dict(orient="records"),
            "intersections": {k: round(v, 2) for k, v in result.model.intersection_baseline.items()},
            "plots": [p.name for p in plot_files],
        }
        config.SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")

    def predict(self, intersection_id: str, timestamp: datetime, vehicles: float,
                previous: Optional[List[float]] = None,
                model_path: Path = config.MODEL_FILE):
        model = TrafficPatternModel.load(model_path)
        if model.baseline_for(intersection_id) is None:
            self.console.warn(f"La intersección '{intersection_id}' no estaba en el entrenamiento; "
                              f"se usa el promedio global. Conocidas: "
                              f"{', '.join(sorted(model.intersection_baseline))}")
        moment = self.clustering.classify(model, intersection_id, timestamp, vehicles, previous)
        self.console.moment(moment)
        return moment
