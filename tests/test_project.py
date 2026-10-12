import tomllib
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

import cbo

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject():
    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] == cbo.__version__


def test_ensure_superuser_is_idempotent(db, capsys):
    from breakfast.models import User

    args = dict(username="boss", email="boss@example.com", password="s3cret")
    call_command("ensure_superuser", **args)
    call_command("ensure_superuser", **args)
    assert User.objects.filter(username="boss", is_superuser=True).count() == 1
    assert User.objects.get(username="boss").check_password("s3cret")
    assert "already exists" in capsys.readouterr().out


def test_ensure_superuser_refuses_empty_password(db):
    with pytest.raises(CommandError):
        call_command("ensure_superuser", username="boss", email="b@example.com", password="")


def test_ensure_schema_is_a_noop_on_sqlite(db, capsys):
    call_command("ensure_schema")
    assert "No PostgreSQL schema" in capsys.readouterr().out


def test_migrations_are_up_to_date(db):
    call_command("makemigrations", "--check", "--dry-run")
