PYTHON ?= python

.PHONY: setup prepare profile train evaluate test lint format api ui run docker-build docker-up clean

setup:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

prepare:
	$(PYTHON) scripts/prepare_raw.py

profile:
	$(PYTHON) scripts/profile_data.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m mypy src/

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

run:
	$(PYTHON) scripts/serve.py

docker-build:
	docker build -t netflix-recommender .

docker-up:
	docker compose up --build

clean:
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree('models', ignore_errors=True); pathlib.Path('models').mkdir(); pathlib.Path('models/.gitkeep').touch()"