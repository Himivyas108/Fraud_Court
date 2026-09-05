FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/reports

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# 7860 matches Hugging Face Spaces' default app_port convention; the app
# itself is just a plain FastAPI/uvicorn service and runs fine on any port.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
