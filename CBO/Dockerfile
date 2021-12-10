# pull official base image
FROM docker-remote.registry.saas.cagip.group.gca/library/python:3.8-slim

ARG POSTGRES_MASTER_DB
ARG POSTGRES_MASTER_USER
ARG POSTGRES_MASTER_PASSWORD
ARG POSTGRES_SCHEMA
ARG POSTGRES_PORT
ARG POSTGRES_URL
ARG PASSWORD
ARG SECRET_KEY
ARG NAME

ENV POSTGRES_MASTER_DB ${POSTGRES_MASTER_DB}
ENV POSTGRES_MASTER_USER ${POSTGRES_MASTER_USER}
ENV POSTGRES_MASTER_PASSWORD ${POSTGRES_MASTER_PASSWORD}
ENV POSTGRES_SCHEMA ${POSTGRES_SCHEMA}
ENV POSTGRES_PORT ${POSTGRES_PORT}
ENV POSTGRES_URL ${POSTGRES_URL}
ENV PGPASSWORD ${POSTGRES_MASTER_PASSWORD}
ENV PASSWORD ${PASSWORD}
ENV SECRET_KEY ${SECRET_KEY}
ENV USER_DJANGO ${NAME}

# copy project
COPY . .

EXPOSE 8000

# install dependencies
RUN apt update \
&& apt install -y libpq-dev gcc git openssh-client postgresql libxml2 \
&& pip install --upgrade pip \
&& pip install pipenv \
&& pipenv run pip install 'setuptools<58' \
&& pipenv install --system --deploy \
&& python manage.py collectstatic --noinput \
&& python manage.py makemigrations

ENTRYPOINT su -m postgres -c "psql -h ${POSTGRES_URL} -p ${POSTGRES_PORT} -U ${POSTGRES_MASTER_USER} -d ${POSTGRES_MASTER_DB} -tc \"CREATE SCHEMA IF NOT EXISTS ${POSTGRES_SCHEMA}\"" \
&& python manage.py migrate \
&& python manage.py ensure_superuser --password=${PASSWORD} --username=${USER_DJANGO} --email=Groupe-recherche-operationnelle.GRO@credit-agricole-sa.fr \
&& gunicorn DjangoSite.wsgi:application --bind 0.0.0.0:8000 --worker-class=gevent --timeout 1000
