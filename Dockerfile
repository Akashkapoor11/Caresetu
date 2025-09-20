# Dockerfile for Render (Streamlit app with Tesseract + Gemini)
FROM python:3.11-slim

# System deps for tesseract and common libs
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

# Create working dir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install python deps
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Set tesseract path env (pytesseract will use this when configured)
ENV TESSERACT_CMD=/usr/bin/tesseract

# Streamlit server settings (use PORT env var provided by Render)
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose port
EXPOSE 8080

# Entrypoint: run streamlit
CMD ["sh", "-c", "streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0"]
