ARG DOCKER_REGISTRY
FROM ${DOCKER_REGISTRY}python:3.8-slim

ENV CSRF_COOKIE_NAME cbocsrftoken
ENV SESSION_COOKIE_NAME cbo
ENV CSRF_COOKIE_DOMAIN .k8s.caas.ca-sa.gca

# copy project
COPY . .

EXPOSE 8000

# install dependencies
RUN apt update \
&& apt install -y libpq-dev gcc git openssh-client postgresql libxml2 \
&& pip install pipenv \
&& pipenv install --system --deploy \
&& python manage.py collectstatic --noinput \
&& python manage.py makemigrations

RUN export APPLICATION_TAG_VERSION=`python -c 'import cbo; print(cbo.__version__)'`

LABEL da.da/cbo.version=$APPLICATION_TAG_VERSION \
      da.da/cbo.contact=Groupe-recherche-operationnelle.GRO@credit-agricole-sa.fr

ENTRYPOINT su -m postgres -c "psql -h ${POSTGRES_URL} -p ${POSTGRES_PORT} -U ${POSTGRES_MASTER_USER} -d ${POSTGRES_MASTER_DB} -tc \"CREATE SCHEMA IF NOT EXISTS ${POSTGRES_SCHEMA}\"" \
&& python manage.py migrate \
&& python manage.py ensure_superuser --password=${PASSWORD} --username=${USERNAME} --email=Groupe-recherche-operationnelle.GRO@credit-agricole-sa.fr \
&& gunicorn DjangoSite.wsgi:application --bind 0.0.0.0:8000 --worker-class=gevent --timeout 1000
