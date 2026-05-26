.PHONY: run install

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

run: $(PYTHON)
	$(PYTHON) -m app

install: $(PYTHON)
	$(PIP) install -r requirements.txt

$(PYTHON):
	@test -x $(PYTHON) || (echo "Virtualenv not found. Run: python3 -m venv .venv && make install" && exit 1)
