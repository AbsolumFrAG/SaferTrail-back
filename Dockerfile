# Dockerfile pour SaferTrail API avec Alpine Linux
FROM python:3.13-alpine

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production \
    PYTHONPATH=/app

# Installation des dépendances système pour les packages géospatiaux
RUN apk update && apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    gdal \
    gdal-dev \
    proj-dev \
    geos-dev \
    curl \
    make \
    cmake \
    && rm -rf /var/cache/apk/*

# Créer l'utilisateur et le répertoire de travail
RUN addgroup -g 1000 safertrail && \
    adduser -D -s /bin/sh -u 1000 -G safertrail safertrail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

# Copier le code source
COPY src/ ./src/
COPY main.py .

# Créer les répertoires nécessaires
RUN mkdir -p data models logs && \
    chown -R safertrail:safertrail /app

# Health check simple
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Port exposé
EXPOSE 8000

# Passer à l'utilisateur non-root
USER safertrail

# Commande par défaut
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "src.api:app"]