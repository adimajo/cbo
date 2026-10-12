"""Sphinx configuration."""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cbo.settings")
os.environ.setdefault("SECRET_KEY", "docs-only")
os.environ.setdefault("TEST_DB", "1")

import django  # noqa: E402

django.setup()

project = "CBO"
author = "Adrien Ehrhardt"
copyright = "2021-2026, Adrien Ehrhardt"
release = version = re.search(r'^__version__ = "(.*)"', (ROOT / "cbo" / "__init__.py").read_text(), re.M).group(1)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "django": ("https://docs.djangoproject.com/en/stable/", "https://docs.djangoproject.com/en/stable/_objects/"),
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {"collapse_navigation": True}
