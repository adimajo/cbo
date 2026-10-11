"""
Django settings for CBO.

Everything deployment-specific is read from environment variables, see the "Configuration" section of the README.
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from django.utils.translation import gettext_lazy as _

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


# ---------------------------------------------------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------------------------------------------------
DEBUG = env_bool("DEBUG", False)

SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("The SECRET_KEY environment variable is required when DEBUG is off.")
    SECRET_KEY = "insecure-development-key"  # nosec B105

ALLOWED_HOSTS = env_list("ALLOWED_HOSTS", "localhost,127.0.0.1")
CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")

# Cookie names are configurable so several deployments can share a parent domain.
SESSION_COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME", "cbo")
CSRF_COOKIE_NAME = os.environ.get("CSRF_COOKIE_NAME", "cbocsrftoken")
CSRF_COOKIE_DOMAIN = os.environ.get("CSRF_COOKIE_DOMAIN") or None
SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = env_bool("SECURE_COOKIES", not DEBUG)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https") if env_bool("BEHIND_PROXY") else None

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "breakfast.apps.BreakfastConfig",
]
AUTH_USER_MODEL = "breakfast.User"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "cbo.urls"
WSGI_APPLICATION = "cbo.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "breakfast.context_processors.site",
            ],
        },
    },
]

# ---------------------------------------------------------------------------------------------------------------------
# Database: PostgreSQL when POSTGRES_MASTER_DB is set, SQLite otherwise (or when TEST_DB=1).
# ---------------------------------------------------------------------------------------------------------------------
POSTGRES_SCHEMA = os.environ.get("POSTGRES_SCHEMA", "")
if env_bool("TEST_DB") or not os.environ.get("POSTGRES_MASTER_DB"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": os.environ.get("SQLITE_PATH", BASE_DIR / "db.sqlite3"),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ["POSTGRES_MASTER_DB"],
            "USER": os.environ.get("POSTGRES_MASTER_USER", ""),
            "PASSWORD": os.environ.get("POSTGRES_MASTER_PASSWORD") or os.environ.get("PGPASSWORD", ""),
            "HOST": os.environ.get("POSTGRES_URL", "localhost"),
            "PORT": os.environ.get("POSTGRES_PORT", "5432"),
            "OPTIONS": {"options": f"-c search_path={POSTGRES_SCHEMA}"} if POSTGRES_SCHEMA else {},
        }
    }
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "home"
LOGOUT_REDIRECT_URL = "home"

# ---------------------------------------------------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------------------------------------------------
LANGUAGE_CODE = os.environ.get("LANGUAGE_CODE", "en")
LANGUAGES = [("en", _("English")), ("fr", _("French"))]
LOCALE_PATHS = [BASE_DIR / "breakfast" / "locale"]
TIME_ZONE = os.environ.get("TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------------------------------------------------
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        if DEBUG
        else "whitenoise.storage.CompressedManifestStaticFilesStorage"
    },
}

# ---------------------------------------------------------------------------------------------------------------------
# E-mail
# ---------------------------------------------------------------------------------------------------------------------
EMAIL_BACKEND_PATH = os.environ.get("EMAIL_BACKEND", "django.core.mail.backends.smtp.EmailBackend")
_smtp_options = {
    "host": os.environ.get("EMAIL_HOST", "localhost"),
    "port": int(os.environ.get("EMAIL_PORT", "25")),
    "username": os.environ.get("EMAIL_HOST_USER", ""),
    "password": os.environ.get("EMAIL_HOST_PASSWORD", ""),
    "use_tls": env_bool("EMAIL_USE_TLS"),
}
MAILERS = {
    "default": {
        "BACKEND": EMAIL_BACKEND_PATH,
        "OPTIONS": _smtp_options if EMAIL_BACKEND_PATH.endswith("smtp.EmailBackend") else {},
    }
}

# ---------------------------------------------------------------------------------------------------------------------
# CBO specific
# ---------------------------------------------------------------------------------------------------------------------
# Shared mailbox used as sender, reply-to, and calendar organiser. The user owning this address is the service account:
# it never takes part in breakfasts.
CBO_ADMIN_EMAIL = os.environ.get("CBO_ADMIN_EMAIL", "cbo@example.com")
DEFAULT_FROM_EMAIL = CBO_ADMIN_EMAIL
CBO_ORGANIZER_NAME = os.environ.get("CBO_ORGANIZER_NAME", "CBO")
CBO_EVENT_LOCATION = os.environ.get("CBO_EVENT_LOCATION", "")
# Team that can be left out of a given breakfast (e.g. a team that only joins on some occasions). Empty disables it.
CBO_OPTIONAL_TEAM = os.environ.get("CBO_OPTIONAL_TEAM", "")
# Drafts that were never confirmed are discarded after this many hours.
CBO_DRAFT_TTL_HOURS = int(os.environ.get("CBO_DRAFT_TTL_HOURS", "24"))
