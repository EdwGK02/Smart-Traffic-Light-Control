"""Vista de consola: imprime resúmenes legibles."""
import pandas as pd


class ConsoleView:
    LINE = "─" * 72

    def header(self, title: str):
        print(f"\n{self.LINE}\n  {title}\n{self.LINE}")

    def info(self, msg: str):
        print(f"• {msg}")

    def success(self, msg: str):
        print(f"✔ {msg}")

    def warn(self, msg: str):
        print(f"⚠ {msg}")

    def data_summary(self, df: pd.DataFrame):
        print(f"  Filas: {len(df):,} | Intersecciones: {df['intersection_id'].nunique()} | "
              f"Rango: {df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}")
        per = df.groupby("intersection_id")["vehicles"].agg(["count", "mean", "max"]).round(1)
        per.columns = ["horas", "prom_vehiculos", "max_vehiculos"]
        print(per.to_string())

    def k_selection(self, scores: pd.DataFrame, chosen: int):
        print("\n  Selección de k (más silhouette = clusters mejor separados):")
        for _, r in scores.iterrows():
            mark = "  ← elegido" if int(r["k"]) == chosen else ""
            bar = "█" * int(max(r["silhouette"], 0) * 40)
            print(f"   k={int(r['k'])}  silhouette={r['silhouette']:.3f}  {bar}{mark}")

    def metrics(self, m: dict):
        print("\n  Métricas del modelo final:")
        print(f"   Silhouette         {m['silhouette']:.3f}   (−1 a 1, mayor es mejor)")
        print(f"   Davies-Bouldin     {m['davies_bouldin']:.3f}   (menor es mejor)")
        print(f"   Calinski-Harabasz  {m['calinski_harabasz']:,.0f}   (mayor es mejor)")

    def profiles(self, prof: pd.DataFrame):
        print("\n  Momentos encontrados (de menor a mayor congestión):")
        show = prof.copy()
        show["vehiculos_prom"] = show["vehiculos_prom"].round(1)
        show["carga_relativa"] = show["carga_relativa"].round(2)
        show["tendencia"] = show["tendencia"].round(3)
        show["pct_fin_semana"] = (show["pct_fin_semana"] * 100).round(0).astype(int).astype(str) + "%"
        show["pct_datos"] = (show["pct_datos"] * 100).round(1).astype(str) + "%"
        cols = ["cluster", "moment", "pct_datos", "vehiculos_prom", "carga_relativa",
                "tendencia", "pct_fin_semana", "horas_tipicas"]
        print(show[cols].to_string(index=False))
        print()

    def moment(self, m):
        self.header("Momento actual de la intersección")
        print(f"  Intersección : {m.intersection_id}")
        print(f"  Fecha/hora   : {m.timestamp:%Y-%m-%d %H:%M} ({['lun','mar','mié','jue','vie','sáb','dom'][m.timestamp.weekday()]})")
        print(f"  Vehículos    : {m.vehicles:g}  (carga relativa {m.relative_load:.2f}×)")
        print(f"  → Momento    : {m.name}  [cluster {m.cluster_id}]")
        if m.distances:
            print("  Distancia a cada momento (menor = más parecido):")
            for name, d in sorted(m.distances.items(), key=lambda x: x[1]):
                print(f"     {name:<28} {d:.2f}")
        print()
