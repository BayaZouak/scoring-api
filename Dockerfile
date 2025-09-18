# Utilise une image de base Python légère
FROM python:3.10-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Variables d’environnement pour éviter les .pyc et activer logs immédiats
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copie le fichier de dépendances en premier pour optimiser le cache Docker
COPY requirements.txt .

# Installe les dépendances système nécessaires (SSL, build, LightGBM / OpenMP)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev \
        gcc \
        build-essential \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie le reste du code dans le conteneur
COPY . .

# Expose le port de l'application
EXPOSE 8000

# Commande de démarrage de l'API
CMD ["uvicorn", "Zouak_Baya_1_API_082025:app", "--host", "0.0.0.0", "--port", "8000"]
