ARG BACKEND_URL=https://phish-nginx/api/analyze-email/
ARG VERIFY_SSL=false
ARG MILTER_PORT=10025
ARG TIMEOUT=600

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    BACKEND_URL=${BACKEND_URL} \
    VERIFY_SSL=${VERIFY_SSL} \
    TIMEOUT=${TIMEOUT} \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libmilter-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pymilter requests tldextract

WORKDIR /app
COPY guard.py .

EXPOSE ${MILTER_PORT}

CMD ["python", "guard.py"]
