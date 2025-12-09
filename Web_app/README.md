## Running with Docker

This project can be built and run using Docker Compose. The provided Dockerfile uses Python 3.13-slim and installs dependencies in a virtual environment. The application exposes port 7860 (the default for Gradio apps).

### Build and Run

```sh
docker compose up --build
```

This will build the image and start the service at [http://localhost:7860](http://localhost:7860).

### Details
- **Python version:** 3.13-slim
- **Exposed port:** 7860
- **No environment variables or volumes are required by default.**
- **No .env file is required unless you uncomment the relevant line in `docker-compose.yml`.**

All necessary model files (`*.pkl`) and the application code (`gr.py`) are included in the image. No additional configuration is needed.