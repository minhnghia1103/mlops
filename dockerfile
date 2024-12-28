#Version Python
FROM python:3.10-slim


# Setup Folder in docker
WORKDIR /app

# Copy requirements.txt file
COPY requirements.txt .

# setup python environment
RUN pip install --no-cache-dir -r requirements.txt

# copy all file in current directory to /app folder in docker
COPY . .

# Expose the port
EXPOSE 8000

# Lệnh để chạy FastAPI với Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

