"""Vista gráfica: genera PNGs para el informe / la exposición."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # sin ventana: solo guarda archivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

import config

DAYS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
# De menor (verde) a mayor congestión (rojo)
PALETTE = ["#2E7D32", "#7CB342", "#FBC02D", "#FB8C00", "#E53935", "#8E24AA", "#5D4037"]


class PlotView:

    def __init__(self, out_dir: Path = config.PLOTS_DIR):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def render_all(self, result) -> List[Path]:
        order = (result.data.groupby("cluster")["relative_load"].mean()
                 .sort_values().index.tolist())
        # Reparte la paleta para que el menor sea siempre verde y el mayor rojo
        pos = np.linspace(0, 4, len(order)).round().astype(int) if len(order) <= 5 \
            else range(len(order))
        colors = {cid: PALETTE[p] for cid, p in zip(order, pos)}
        return [
            self.k_selection(result.k_scores, result.metrics["k"]),
            self.hourly_profile(result.data, order, colors),
            self.week_heatmap(result.data, order, colors),
            self.pca_scatter(result, order, colors),
        ]

    # 1. Codo + silhouette
    def k_selection(self, scores: pd.DataFrame, chosen: int) -> Path:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(scores["k"], scores["inertia"], "o-", color="#1565C0", label="Inercia")
        ax1.set_xlabel("Número de clusters (k)")
        ax1.set_ylabel("Inercia (codo)", color="#1565C0")
        ax2 = ax1.twinx()
        ax2.plot(scores["k"], scores["silhouette"], "s--", color="#E53935", label="Silhouette")
        ax2.set_ylabel("Silhouette", color="#E53935")
        ax1.axvline(chosen, color="gray", ls=":", lw=1)
        ax1.set_title(f"Selección de k (elegido: {chosen})")
        fig.tight_layout()
        return self._save(fig, "01_seleccion_k.png")

    # 2. Vehículos promedio por hora, coloreado por momento dominante
    def hourly_profile(self, data: pd.DataFrame, order, colors) -> Path:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for cid in order:
            sub = data[data["cluster"] == cid]
            counts = sub.groupby("hour").size().reindex(range(24), fill_value=0)
            ax.bar(range(24), counts, color=colors[cid], alpha=0.85,
                   label=sub["moment"].iloc[0],
                   bottom=self._stack_bottom(data, order, cid))
        ax.set_xticks(range(24))
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Nº de horas observadas")
        ax.set_title("¿En qué momento está la intersección a cada hora?")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()
        return self._save(fig, "02_momentos_por_hora.png")

    @staticmethod
    def _stack_bottom(data, order, cid):
        idx = order.index(cid)
        bottom = np.zeros(24)
        for prev in order[:idx]:
            bottom += data[data["cluster"] == prev].groupby("hour").size().reindex(range(24), fill_value=0).values
        return bottom

    # 3. Mapa semanal hora × día con el momento más frecuente
    def week_heatmap(self, data: pd.DataFrame, order, colors) -> Path:
        rank = {cid: i for i, cid in enumerate(order)}
        mode = (data.groupby(["day_of_week", "hour"])["cluster"]
                .agg(lambda s: s.value_counts().idxmax())
                .unstack().reindex(index=range(7), columns=range(24)))
        grid = mode.apply(lambda col: col.map(rank))
        cmap = ListedColormap([colors[c] for c in order])
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.imshow(grid.values, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(order) - 0.5)
        ax.set_yticks(range(7), DAYS)
        ax.set_xticks(range(24), [f"{h}" for h in range(24)])
        ax.set_xlabel("Hora del día")
        ax.set_title("Momento más frecuente por día y hora")
        names = data.drop_duplicates("cluster").set_index("cluster")["moment"]
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in order]
        ax.legend(handles, [names[c] for c in order], bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=8)
        fig.tight_layout()
        return self._save(fig, "03_mapa_semanal.png")

    # 4. Proyección 2D de los clusters
    def pca_scatter(self, result, order, colors) -> Path:
        X = result.model.scaled(result.data)
        n = min(6000, len(X))
        idx = np.random.default_rng(0).choice(len(X), n, replace=False)
        pts = PCA(n_components=2, random_state=0).fit_transform(X[idx])
        labels = result.data["cluster"].values[idx]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for cid in order:
            m = labels == cid
            ax.scatter(pts[m, 0], pts[m, 1], s=6, alpha=0.5, color=colors[cid],
                       label=result.model.name_of(cid))
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.set_title("Clusters en 2D (PCA)")
        ax.legend(markerscale=3, fontsize=8)
        fig.tight_layout()
        return self._save(fig, "04_clusters_pca.png")

    def _save(self, fig, name: str) -> Path:
        path = self.out / name
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
