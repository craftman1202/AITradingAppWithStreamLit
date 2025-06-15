# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

# Copy application files
COPY app.py .
COPY feature_utils.py .
COPY best_rf_model_ni225_diff_maxitrade.pkl .
COPY NI225_RF_MetaLabel.pkl .
COPY requirements.txt .
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default to exposing Streamlit's local dev port
EXPOSE 8501

# Cloud Run expects the app to listen on $PORT (default 8080).
# Streamlit will read the PORT environment variable, defaulting to 8501 for local dev.
ENV PORT=8080

# Use bash -c to evaluate the env variable for port
CMD bash -c "streamlit run app.py --server.port=\${PORT:-8501} --server.address=0.0.0.0"
