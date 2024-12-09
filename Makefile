# Variables
APP_NAME = app.py
VENV_DIR = venv
REQ_FILE = requirements.txt
PYTHON = python3

# Default target
all: run

# Create virtual environment
$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)

# Install dependencies
install: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE)

# Run the Flask application
run: install
	FLASK_APP=$(APP_NAME) FLASK_ENV=development $(VENV_DIR)/bin/flask run

# Format Python code (optional, if you want to use tools like black)
format:
	$(VENV_DIR)/bin/pip install black
	$(VENV_DIR)/bin/black $(APP_NAME)

# Lint Python code (optional, if you want to use tools like pylint)
lint:
	$(VENV_DIR)/bin/pip install pylint
	$(VENV_DIR)/bin/pylint $(APP_NAME)

# Clean up: remove virtual environment and temporary files
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Help: display targets
help:
	@echo "Usage:"
	@echo "  make run      - Run the Flask app"
	@echo "  make install  - Install dependencies"
	@echo "  make format   - Format code with Black (optional)"
	@echo "  make lint     - Lint code with pylint (optional)"
	@echo "  make clean    - Remove virtual environment and __pycache__"

