"""
Smart Traffic Light Control — Etapa 1: agrupar patrones de tráfico.

Uso:
  python main.py train                                  # Kaggle, k automático
  python main.py train --source bogota --data data/raw/mi_archivo.csv
  python main.py train --source both --k 5
  python main.py predict --junction J1 --datetime "2017-06-30 08:00" --vehicles 60 --prev 35 48
  python main.py web                                    # página en http://127.0.0.1:5000
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import config
from app.controllers import ClusteringController


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smart Traffic Light Control · Etapa 1 (clustering)")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Entrena el modelo de patrones de tráfico")
    t.add_argument("--source", choices=["kaggle", "bogota", "both"], default="kaggle")
    t.add_argument("--data", type=Path, default=None, help="Ruta al CSV (opcional)")
    t.add_argument("--k", type=int, default=None, help="Número de clusters (por defecto: automático)")
    t.add_argument("--no-plots", action="store_true", help="No generar gráficas")
    t.add_argument("--model", type=Path, default=config.MODEL_FILE)

    pr = sub.add_parser("predict", help="Clasifica el momento actual de una intersección")
    pr.add_argument("--junction", required=True, help="ID de la intersección (ej. J1, BOG_123)")
    pr.add_argument("--datetime", required=True, help='Fecha y hora, ej. "2017-06-30 08:00"')
    pr.add_argument("--vehicles", type=float, required=True, help="Vehículos en esa hora")
    pr.add_argument("--prev", type=float, nargs="*", default=[],
                    help="Conteos de horas anteriores (de más antigua a más reciente)")
    pr.add_argument("--model", type=Path, default=config.MODEL_FILE)

    w = sub.add_parser("web", help="Levanta la página web (Flask)")
    w.add_argument("--host", default=config.FLASK_HOST)
    w.add_argument("--port", type=int, default=config.FLASK_PORT)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "web":
        from app.web import create_app
        create_app().run(host=args.host, port=args.port, debug=True)
        return 0

    controller = ClusteringController()
    try:
        if args.command == "train":
            controller.train(source=args.source, data_path=args.data, k=args.k,
                             model_path=args.model, make_plots=not args.no_plots)
        elif args.command == "predict":
            controller.predict(args.junction, datetime.fromisoformat(args.datetime),
                               args.vehicles, args.prev, model_path=args.model)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✖ {e}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
