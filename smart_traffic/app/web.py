"""
Fábrica de la aplicación Flask.

Las plantillas HTML y los archivos estáticos viven en app/views/ porque son
parte de la capa VIEW; las rutas viven en app/controllers/web_controller.py.
"""
from pathlib import Path

from flask import Flask

import config

VIEWS_DIR = Path(__file__).resolve().parent / "views"


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(VIEWS_DIR / "templates"),
        static_folder=str(VIEWS_DIR / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_AS_ASCII"] = False

    from app.controllers.web_controller import web
    app.register_blueprint(web)
    return app
