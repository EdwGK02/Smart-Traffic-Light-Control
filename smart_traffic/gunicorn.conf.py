"""
Configuración de Gunicorn.

    gunicorn -c gunicorn.conf.py wsgi:app

Nota: Gunicorn funciona en Linux y macOS (no en Windows).
En Windows usen `python server.py` para desarrollo, o WSL para probar Gunicorn.
"""
import os

# Dirección: en un servidor/Docker usa 0.0.0.0 para aceptar conexiones externas
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Procesos y hilos. 2 workers bastan para el proyecto; los hilos
# permiten atender la página mientras otro usuario entrena.
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
worker_class = "gthread"

# El entrenamiento puede tardar con el dataset completo: damos 5 minutos
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))
graceful_timeout = 30
keepalive = 5

# Logs a la consola
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Carga la app antes de crear los workers (arranque más rápido, menos memoria)
preload_app = True
