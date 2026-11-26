"""Production entry point: `gunicorn wsgi:app`."""
from lad.api import create_app

app = create_app()
