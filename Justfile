# Setup local
install-hooks:
    uv run pre-commit install

setup:
    @command -v uv >/dev/null 2>&1 && echo "✅ uv already installed" || (echo "🔧 Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)

    uv sync --all-groups --cache-dir .uv_cache
    echo "Setup done. Run 'source .venv/bin/activate' to activate the virtual environment."

# Format code
lint:
    ruff check .

fmt:
    uv run ruff check --fix
    uv run isort ./app

# Server
dependency:
    # Agents
    uv export --only-group text_embedding -o app/text_embedding/requirements.txt


stop:
    docker-compose -p rag -f docker/docker-compose.yml down

start: stop
    docker-compose -p rag -f docker/docker-compose.yml up -d
