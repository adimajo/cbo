#!/bin/sh
# Prepare the database, make sure an administrator exists, then serve the application.
set -eu

python manage.py ensure_schema
python manage.py migrate --noinput

if [ -n "${CBO_ADMIN_USERNAME:-}" ] && [ -n "${CBO_ADMIN_PASSWORD:-}" ]; then
    python manage.py ensure_superuser \
        --username "$CBO_ADMIN_USERNAME" \
        --password "$CBO_ADMIN_PASSWORD" \
        --email "${CBO_ADMIN_EMAIL:-cbo@example.com}"
fi

exec gunicorn cbo.wsgi:application --bind 0.0.0.0:8000 --workers "${WEB_CONCURRENCY:-2}" --timeout 120 \
    --access-logfile -
