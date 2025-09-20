FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

COPY . /app
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8080
CMD ["sh", "-c", "streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0"]
