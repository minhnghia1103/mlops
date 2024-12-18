FROM python:3.8-slim


# Setup Folder in docker
WORKDIR /app

# Copy requirements.txt file
COPY requirements.txt .

# setup python environment
RUN pip install --no-cache-dir -r requirements.txt

# copy all file in current directory to /app folder in docker
COPY . .

# run the main.py file
CMD ["python", "app/main.py"]
