# config.py
import os
from datetime import timedelta

class Config:
    # Secret key for sessions
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-this"

    # Database credentials (correct env var names)
    DB_USER = os.environ.get("DB_USER") or "flaskuser"
    DB_PASS = os.environ.get("DB_PASS") or "FlaskPass123!"
    DB_HOST = os.environ.get("DB_HOST") or "127.0.0.1"
    DB_NAME = os.environ.get("DB_NAME") or "institution_portal"

    # SQLAlchemy connection string
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Cookie/session security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False  # Set to True ONLY on HTTPS

    PERMANENT_SESSION_LIFETIME = timedelta(hours=4)
