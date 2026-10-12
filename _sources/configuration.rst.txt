Configuration
=============

CBO is configured exclusively through environment variables.

Core
----

``SECRET_KEY``
    Django secret key. Required unless ``DEBUG`` is on.
``DEBUG``
    ``1`` to enable Django's debug mode (default: off).
``ALLOWED_HOSTS``
    Comma-separated host names (default: ``localhost,127.0.0.1``).
``CSRF_TRUSTED_ORIGINS``
    Comma-separated origins, e.g. ``https://cbo.example.com``.
``BEHIND_PROXY``
    ``1`` when a TLS-terminating proxy sets ``X-Forwarded-Proto``.
``SECURE_COOKIES``
    Force the ``Secure`` flag on cookies (default: on unless ``DEBUG``).
``SESSION_COOKIE_NAME``, ``CSRF_COOKIE_NAME``, ``CSRF_COOKIE_DOMAIN``
    Cookie settings, useful when several deployments share a parent domain.
``LANGUAGE_CODE``
    Default interface language, ``en`` or ``fr`` (default: ``en``). Users can override it from the header.
``TIME_ZONE``
    Time zone used to display and enter dates (default: ``UTC``).

Database
--------

SQLite is used unless ``POSTGRES_MASTER_DB`` is set (or ``TEST_DB=1`` forces SQLite).

``SQLITE_PATH``
    Location of the SQLite file (default: ``db.sqlite3`` next to ``manage.py``).
``POSTGRES_MASTER_DB``, ``POSTGRES_MASTER_USER``, ``POSTGRES_MASTER_PASSWORD`` (or ``PGPASSWORD``), ``POSTGRES_URL``, ``POSTGRES_PORT``
    PostgreSQL connection.
``POSTGRES_SCHEMA``
    Schema holding the tables. ``manage.py ensure_schema`` creates it when missing (the Docker image does so).

E-mail
------

``EMAIL_HOST``, ``EMAIL_PORT``, ``EMAIL_HOST_USER``, ``EMAIL_HOST_PASSWORD``, ``EMAIL_USE_TLS``
    SMTP server.
``EMAIL_BACKEND``
    Backend class path, e.g. ``django.core.mail.backends.console.EmailBackend`` to print mails while developing.

Application
-----------

``CBO_ADMIN_EMAIL``
    Shared mailbox used as sender, reply-to, copy and calendar organiser. The user with this address is a service
    account and never takes part in breakfasts.
``CBO_ORGANIZER_NAME``
    Display name of the calendar organiser (default: ``CBO``).
``CBO_EVENT_LOCATION``
    Location written in the calendar invitation (default: empty).
``CBO_OPTIONAL_TEAM``
    Name of a team (the ``team`` field of users) that can be left out of a given breakfast. When set, the new
    breakfast page shows a switch for it.
``CBO_DRAFT_TTL_HOURS``
    Unconfirmed drafts older than this are discarded when a new draft is created (default: 24).

Docker only
-----------

``CBO_ADMIN_USERNAME``, ``CBO_ADMIN_PASSWORD``
    When both are set, the entrypoint creates this superuser if it does not exist.
``WEB_CONCURRENCY``
    Number of gunicorn workers (default: 2).
