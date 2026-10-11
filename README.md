[![CI](https://github.com/adimajo/cbo/actions/workflows/ci.yml/badge.svg)](https://github.com/adimajo/cbo/actions/workflows/ci.yml)
[![pipeline status](https://gitlab.com/adimajo/cbo/badges/master/pipeline.svg)](https://gitlab.com/adimajo/cbo/-/commits/master)
[![Coverage status](https://codecov.io/gh/adimajo/cbo/branch/master/graph/badge.svg)](https://codecov.io/github/adimajo/cbo?branch=master)

# CBO, the Chief Breakfast Officer

CBO is a [Django](https://www.djangoproject.com/) web application that organises team breakfasts fairly.

* Everybody has a **points balance**. Attending a breakfast costs one point; hosting earns the points paid by the
  participants (+50% when the food is home-made).
* Hosts are **drawn at random**, the lower someone's balance the likelier they are picked.
* Hosts and participants are **notified by e-mail**, with a calendar invitation attached.
* The interface is available in **English and French**; each user switches language from the header.

Documentation: <https://adimajo.github.io/cbo/>

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
uv run python scripts/compile_messages.py     # build the translation catalogues
export DEBUG=1 EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
uv run python manage.py migrate
uv run python manage.py createsuperuser
uv run python manage.py runserver
```

Open <http://localhost:8000/>, sign in, add the team members in the administration (<http://localhost:8000/admin/>,
*Users*: set e-mail, team and arrival date), then organise a breakfast from *New breakfast*. Without further
configuration the app uses SQLite and prints e-mails to the console.

## Configuration

Everything is configured through environment variables; the full list is in the
[documentation](https://adimajo.github.io/cbo/configuration.html). The essentials:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Django secret key (required unless `DEBUG=1`) |
| `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS` | Comma-separated hosts / origins the site is served from |
| `POSTGRES_MASTER_DB`, `POSTGRES_MASTER_USER`, `POSTGRES_MASTER_PASSWORD`, `POSTGRES_URL`, `POSTGRES_PORT`, `POSTGRES_SCHEMA` | PostgreSQL connection; SQLite is used when `POSTGRES_MASTER_DB` is unset |
| `EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD`, `EMAIL_USE_TLS` | SMTP server |
| `CBO_ADMIN_EMAIL` | Shared mailbox used as sender and calendar organiser |
| `CBO_OPTIONAL_TEAM` | A team that can be left out of a given breakfast |
| `LANGUAGE_CODE` | Default language, `en` or `fr` |
| `TIME_ZONE` | Time zone for dates (default `UTC`) |

## Docker

```bash
docker build -t cbo .
docker run -p 8000:8000 -v cbo-data:/data \
    -e SECRET_KEY=change-me -e ALLOWED_HOSTS=localhost -e SECURE_COOKIES=0 \
    -e CBO_ADMIN_USERNAME=admin -e CBO_ADMIN_PASSWORD=change-me \
    cbo
```

The entrypoint creates the PostgreSQL schema when needed, applies the migrations, creates the administrator when
`CBO_ADMIN_USERNAME` and `CBO_ADMIN_PASSWORD` are set, and starts gunicorn on port 8000. Images are published to
`ghcr.io/adimajo/cbo`.

## Development

```bash
uv sync
uv run python scripts/compile_messages.py
uv run pytest                                  # tests
uv run ruff check . && uv run ruff format .    # lint and format
uv run bandit -r cbo breakfast                 # security linter
uv run --group docs make -C docs html          # documentation
```

### Translations

User-facing strings go through Django's i18n. The catalogues are in `breakfast/locale/<language>/LC_MESSAGES/`.
To add or refresh a language run `python manage.py makemessages -l <code>` (needs GNU gettext), edit the `.po`
file, compile with `python manage.py compilemessages` (or `python scripts/compile_messages.py`, which only needs
Python) and list the language in `LANGUAGES` in `cbo/settings.py`. Compiled `.mo` files are not committed.

## Upgrading from 1.x

Version 2 is a rewrite and **the database schema changed**: the app and the models were renamed to English
(`Dej` → `breakfast`, `PetitDej` → `Breakfast`, `nombre_points` → `points`, …) and the migrations are now part of the
repository (1.x generated them when building the image). A 1.x database cannot be migrated in place: start from an
empty database, then recreate the users in the administration. Other changes to be aware of:

* the superuser variables are now `CBO_ADMIN_USERNAME` / `CBO_ADMIN_PASSWORD` (`USERNAME` collided with the one the
  operating system sets);
* `DEBUG` is off by default and `SECRET_KEY` is required;
* host names, cookie names, the shared mailbox and the optional team were hard-coded and are now configurable;
* a breakfast only counts towards points once confirmed, so abandoned drafts no longer cost anybody a point.
