# 1. Use a lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (this makes rebuilding faster)
COPY requirements.txt .

# 4. Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy EVERY folder and file from your computer into the container
# This includes Models/, Input_Data/, Files/, etc.
COPY . .

# 6. CRITICAL: Hugging Face runs on Linux. 
# We need to give the server permission to write/save CSVs in your folders.
RUN chmod -R 777 /app

# 7. Tell the container to run your FastAPI gateway
# Port 7860 is the specific port Hugging Face looks for
CMD ["python", "main.py"]