# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies first (caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]