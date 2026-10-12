ARG DOCKER_REGISTRY=
ARG PYTHON_VERSION=3.13

# ---- build: resolve dependencies, compile translations, collect static files ----------------------------------------
FROM ${DOCKER_REGISTRY}python:${PYTHON_VERSION}-slim AS build
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends gettext && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
RUN export SECRET_KEY=build-only \
    && /opt/venv/bin/python manage.py compilemessages \
    && /opt/venv/bin/python manage.py collectstatic --noinput

# ---- runtime ---------------------------------------------------------------------------------------------------------
FROM ${DOCKER_REGISTRY}python:${PYTHON_VERSION}-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --system --create-home --uid 10001 cbo && mkdir /data && chown cbo /data
ENV SQLITE_PATH=/data/db.sqlite3
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=cbo /app /app
USER cbo
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
