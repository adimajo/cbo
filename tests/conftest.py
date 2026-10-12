import sys
from datetime import date, datetime, time
from pathlib import Path

import pytest
from django.utils import timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compile_messages import compile_messages  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def compiled_translations():
    compile_messages()


@pytest.fixture(autouse=True)
def plain_static_storage(settings):
    settings.PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]  # fast, tests only
    settings.STORAGES = {
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }


@pytest.fixture
def make_user(db):
    from breakfast.models import User

    def _make(username, **kwargs):
        kwargs.setdefault("email", f"{username}@example.com")
        kwargs.setdefault("arrival_date", date(2020, 1, 1))
        return User.objects.create_user(username=username, password="pw", **kwargs)

    return _make


@pytest.fixture
def team(make_user):
    return [make_user(name) for name in ("alice", "bob", "carol", "dave")]


@pytest.fixture
def make_breakfast(db):
    from breakfast.models import Breakfast, Host

    def _make(day=date(2024, 5, 14), hosts=(), absentees=(), confirmed=True, homemade=(), **kwargs):
        breakfast = Breakfast.objects.create(
            date=timezone.make_aware(datetime.combine(day, time(9, 0))),
            confirmed_at=timezone.now() if confirmed else None,
            **kwargs,
        )
        for host in hosts:
            Host.objects.create(breakfast=breakfast, user=host, homemade=host in homemade)
        breakfast.absentees.set(absentees)
        return breakfast

    return _make


@pytest.fixture
def logged_client(client, team):
    client.force_login(team[0])
    return client
