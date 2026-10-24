"""
Punto de entrada WSGI para Gunicorn (producción).

    gunicorn -c gunicorn.conf.py wsgi:app
"""
from app.web import create_app

app = create_app()
