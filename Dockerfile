# Use Ubuntu with Python (easier ODBC installation)
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and ODBC driver
RUN apt-get update && \
    apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    gnupg \
    unixodbc \
    unixodbc-dev && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Update requirements to use pyodbc
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Python application files
COPY notebook/LSTM_Scheduler.py .
COPY notebook/LSTM_Prediction_pipeline_compaware.py .
COPY notebook/LSTM_Prediction_pipeline_2.py .
COPY notebook/LSTM_Database_writer2.py .
COPY models/ ./models/
# Create directories
RUN mkdir -p /app/logs /app/checkpoints

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Healthcheck
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import os; exit(0 if os.path.exists('lstm_scheduler.log') else 1)"

# Run scheduler
CMD ["python3", "LSTM_Scheduler.py", "--interval", "10", "--min-records", "100"]