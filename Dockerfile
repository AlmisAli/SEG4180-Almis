FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

# Serve with Waitress (production-friendly)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]