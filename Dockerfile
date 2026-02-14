# -------- Base Image --------
FROM python:3.10-slim

# -------- System Dependencies --------
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

# -------- Environment Settings --------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -------- Working Directory --------
WORKDIR /app

# -------- Install Python Dependencies --------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# -------- Copy Application Code --------
COPY . /app

# -------- Expose Port --------
EXPOSE 8000

# -------- Run Application --------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]