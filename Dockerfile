# 1. Use the Python 3.9 slim image as the base.
FROM python:3.9-slim

# 2. Set the working directory to /app.
WORKDIR /app

# 3. Install required OS packages.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    swig \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libpulse-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements.txt and install Python libraries.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Copy all project files into the container.
COPY . .

# 6. Run the Flask app.
CMD ["python", "app.py"]
