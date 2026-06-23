# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Copy the rest of the application files
COPY . .

# Generate dataset and train/evaluate models at build time to bake artifacts in
RUN python src/generate_dataset.py
RUN python src/train_evaluate.py

# Expose service port
EXPOSE 5000

# Start app using gunicorn production web server
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 "app:create_app()"
