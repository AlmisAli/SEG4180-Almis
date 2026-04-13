## Run locally (Docker)

```bash
docker build -t YOURDOCKERUSER/model-service:latest .
docker run --rm -p 5000:5000 YOURDOCKERUSER/model-service:latest