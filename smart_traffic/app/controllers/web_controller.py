"""
WebController: rutas de Flask (Blueprint).

Recibe las peticiones de la página web, usa el ClusteringController para
entrenar/predecir y devuelve HTML (la vista) o JSON (para el JavaScript).

Rutas:
  GET  /                    -> página principal (templates/index.html)
  GET  /api/status          -> ¿hay modelo? ¿hay datos? resumen del último entrenamiento
  POST /api/upload          -> sube un CSV a data/raw/
  POST /api/train           -> entrena el modelo
  POST /api/predict         -> clasifica el momento de una intersección
  GET  /plots/<archivo>     -> sirve las gráficas generadas
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import config
from app.controllers.clustering_controller import ClusteringController

web = Blueprint("web", __name__)
_controller = ClusteringController()
_train_lock = threading.Lock()   # evita dos entrenamientos al mismo tiempo

SOURCE_FILES = {"kaggle": config.KAGGLE_FILE, "bogota": config.BOGOTA_FILE}


def _error(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _summary():
    if config.SUMMARY_FILE.exists():
        return json.loads(config.SUMMARY_FILE.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------- página
@web.get("/")
def index():
    return render_template("index.html", project="Smart Traffic Light Control")


# ------------------------------------------------------------------- API
@web.get("/api/status")
def status():
    return jsonify({
        "ok": True,
        "model_ready": config.MODEL_FILE.exists(),
        "datasets": {name: path.exists() for name, path in SOURCE_FILES.items()},
        "summary": _summary(),
    })


@web.post("/api/upload")
def upload():
    source = request.form.get("source", "kaggle")
    file = request.files.get("file")
    if source not in SOURCE_FILES:
        return _error("Fuente inválida: usa kaggle o bogota")
    if not file or not file.filename:
        return _error("No se recibió ningún archivo")
    name = secure_filename(file.filename)
    if not name.lower().endswith((".csv", ".xlsx", ".xls")):
        return _error("El archivo debe ser .csv, .xlsx o .xls")

    dest = SOURCE_FILES[source]
    if name.lower().endswith((".xlsx", ".xls")):
        dest = dest.with_suffix(Path(name).suffix.lower())
    dest.parent.mkdir(parents=True, exist_ok=True)
    file.save(dest)
    return jsonify({"ok": True, "saved_as": dest.name, "source": source})


@web.post("/api/train")
def train():
    body = request.get_json(silent=True) or {}
    source = body.get("source", "kaggle")
    k = body.get("k")
    k = int(k) if k not in (None, "", "auto") else None
    if k is not None and not 2 <= k <= 10:
        return _error("k debe estar entre 2 y 10")

    if not _train_lock.acquire(blocking=False):
        return _error("Ya hay un entrenamiento en curso, espera a que termine", 409)
    try:
        data_path = None
        if source == "bogota":
            # Acepta el archivo subido en .csv o en Excel
            for ext in (".csv", ".xlsx", ".xls"):
                p = config.BOGOTA_FILE.with_suffix(ext)
                if p.exists():
                    data_path = p
                    break
        _controller.train(source=source, data_path=data_path, k=k)
        return jsonify({"ok": True, "summary": _summary()})
    except (FileNotFoundError, ValueError) as e:
        return _error(str(e))
    finally:
        _train_lock.release()


@web.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        junction = str(body["junction"]).strip()
        when = datetime.fromisoformat(str(body["datetime"]))
        vehicles = float(body["vehicles"])
        prev_raw = body.get("prev") or []
        if isinstance(prev_raw, str):
            prev_raw = [x for x in prev_raw.replace(";", ",").split(",") if x.strip()]
        previous = [float(x) for x in prev_raw]
    except (KeyError, ValueError, TypeError):
        return _error("Revisa los campos: intersección, fecha/hora, vehículos y horas anteriores")
    if vehicles < 0 or any(p < 0 for p in previous):
        return _error("Los conteos de vehículos no pueden ser negativos")

    try:
        moment = _controller.predict(junction, when, vehicles, previous)
    except FileNotFoundError as e:
        return _error(str(e), 404)

    result = moment.to_dict()
    result["distances"] = {k: round(v, 3) for k, v in (moment.distances or {}).items()}
    return jsonify({"ok": True, "result": result})


@web.get("/plots/<path:filename>")
def plots(filename):
    return send_from_directory(config.PLOTS_DIR, filename)
