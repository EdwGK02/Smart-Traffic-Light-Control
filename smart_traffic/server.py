"""
Levanta la página web del proyecto.

    python server.py
    -> abre http://127.0.0.1:5000
"""
import config
from app.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True)
