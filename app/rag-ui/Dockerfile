FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY . .

RUN uv venv && \
    uv pip install --upgrade pip && \
    uv pip install -r requirements.txt --no-cache


CMD [".venv/bin/python", "-m", "streamlit", "run", "Home.py", "--server.port", "3000"]
