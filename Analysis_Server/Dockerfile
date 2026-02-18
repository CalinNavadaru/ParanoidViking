FROM python:3.10-slim

ARG PORT=8000
ENV PORT=${PORT} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app
COPY requirements.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements.txt \
 && mkdir -p $NLTK_DATA \
 && python -c "import nltk; nltk.download('stopwords', download_dir='$NLTK_DATA')" \
 && apt-get purge -y build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE ${PORT}

ENTRYPOINT ["sh", "-c"]
CMD ["gunicorn Backend.wsgi:application --bind 0.0.0.0:${PORT}"]
