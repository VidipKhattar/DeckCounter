FROM python:3.11-slim

# opencv-python needs these system libs even when there's no display attached.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

EXPOSE 7860

# Render assigns the port to listen on via $PORT; other hosts (or a plain
# `docker run` with nothing set) fall back to 7860. Shell form (not exec
# form) is required here so the variable actually gets expanded.
CMD uvicorn src.web.server:app --host 0.0.0.0 --port ${PORT:-7860}
